use std::io::Cursor;

fn main() -> anyhow::Result<()> {
    let mut el = wasm_canvas::EventLoop::new();

    let bytes = include_bytes!("../images/tree.jpg");

    let img = image::io::Reader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .decode()?
        .into_rgba8();

    let width = img.width();
    let height = img.height();

    let mut picture = wasm_canvas::Picture::new(img.into_raw().into_boxed_slice(), width, height);

    let mut metrics = wasm_canvas::Metrics::default();

    for _ in 0..300 {
        el.tick(&mut picture, &mut metrics);
    }
    eprintln!("{:?}", &metrics);

    Ok(())
}
