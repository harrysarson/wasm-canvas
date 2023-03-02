#![feature(array_chunks)]

use gloo_timers::future::TimeoutFuture;

use std::cell::RefCell;

use std::collections::HashSet;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use std::io::Cursor;
use std::iter::zip;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use web_sys::console;
use web_sys::ImageData;

#[cfg(test)]
use proptest_derive::Arbitrary;

#[derive(Debug, Default)]
pub struct Metrics {
    _null_swaps: usize,
    _swaps: usize,
}

#[derive(PartialEq, PartialOrd, Ord, Eq, Debug)]
#[cfg(test)]
#[derive(Arbitrary)]
struct CandidatePixel {
    /// this goes first for partial ord
    difference_to_neighbors: u8,

    color: [u8; 3],
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn ord(lhs: CandidatePixel, rhs: CandidatePixel) {
           if lhs.difference_to_neighbors < rhs.difference_to_neighbors {
            assert!(lhs < rhs);
           } else if lhs.difference_to_neighbors > rhs.difference_to_neighbors {
            assert!(lhs > rhs);
           }
        }
    }
}

pub struct Picture {
    data: Box<[u8]>,
    width: u32,
    height: u32,
}

impl Picture {
    pub fn new(data: Box<[u8]>, width: u32, height: u32) -> Self { Self { data, width, height } }

    fn get_pix(&self, r: u32, c: u32) -> [u8; 3] {
        let pix_idx = r * self.width + c;
        let data_idx: usize = (pix_idx * 4).try_into().unwrap();
        self.data[data_idx..][0..3].try_into().unwrap()
    }
}

#[derive(Default, Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
struct Coordinate(u32, u32);

struct Candidates {
    /// Maps pixels coordinate to its cost at time of insertion and color. Must be kept up to date.
    color: BTreeMap<Coordinate, (u8, [u8; 3])>,

    /// Maps pixels cost at time of insertion to its location
    cost: [HashSet<Coordinate>; 256],

    scratch: Vec<u8>,
}

impl Candidates {
    fn new() -> Self {
        return Self {
            color: Default::default(),
            cost: [(); 256].map(|_| Default::default()),
            scratch: Default::default(),
        };
    }

    fn add(&mut self, picture: &Picture, r: u32, c: u32) {
        let color = picture.get_pix(r, c);
        let coor = Coordinate(r, c);
        let new_cost = get_cost(picture, color, r, c);

        match self.color.entry(coor) {
            Entry::Vacant(v) => {
                v.insert((new_cost, color));

                assert!(self.cost[new_cost as usize].insert(coor));
            }
            Entry::Occupied(mut o) => {
                let (cost_at_time_of_insertion, color) = o.get_mut();

                assert!(color == color);
                assert!(self.cost[*cost_at_time_of_insertion as usize].remove(&coor));

                assert!(self.cost[new_cost as usize].insert(coor));
                *cost_at_time_of_insertion = new_cost;
            }
        };
    }

    fn pop_highest_cost(&mut self) -> Option<(Coordinate, [u8; 3])> {
        let highest_cost = {
            let mut highest_cost = 255;

            loop {
                if !self.cost[highest_cost].is_empty() {
                    break Some(highest_cost);
                }
                if highest_cost == 0 {
                    break None;
                }
                highest_cost -= 1;
            }
        };

        highest_cost.map(|highest_cost| {
            let coor = *self.cost[highest_cost]
                .iter().next()
                .expect("this is not empty");

            assert!(self.cost[highest_cost].remove(&coor));

            let color = self
                .color
                .remove(&coor)
                .expect("must be in map as we found cost")
                .1;

            (coor, color)
        })
    }

    fn pop_all_lowest_cost(&mut self) {
        let lowest_cost = {
            let mut lowest_cost = 0;

            loop {
                if !self.cost[lowest_cost].is_empty() {
                    break Some(lowest_cost);
                }
                if lowest_cost == 255 {
                    break None;
                }
                lowest_cost += 1;
            }
        };

        lowest_cost.map(|lowest_cost| {
            for coor in &self.cost[lowest_cost] {
                self.color
                    .remove(coor)
                    .expect("must be in map as we found cost");
            }
            self.cost[lowest_cost].clear();
        });
    }

    fn len(&self) -> usize {
        let len = self.color.len();

        if cfg!(debug_assertions) {
            let mut cost_len = 0;
            for costs in &self.cost {
                cost_len += costs.len();
            }
            assert_eq!(cost_len, len);
        }

        return len;
    }

    fn remove_candidate_minimising<T: PartialOrd>(
        &mut self,
        mut check_count: usize,
        f: impl Fn([u8; 3], Coordinate) -> T,
    ) -> Option<Coordinate> {
        self.scratch.reserve(self.color.len());
        self.scratch.clear();
        let mut minimising_coor = None;
        let mut current_min = None;

        'outer: for cost in (0..256).rev() {
            for coor in &self.cost[cost] {
                if check_count == 0 {
                    break 'outer;
                }
                check_count -= 1;

                let (_cost, color) = self.color[coor];
                let val = f(color, *coor);
                if current_min
                    .as_ref()
                    .map_or(true, |current_min| &val <= current_min)
                {
                    current_min = Some(val);
                    minimising_coor = Some(*coor);
                }
            }
        }

        minimising_coor.map(|minimising_coor| {
            let (cost_at_time_of_insertion, _color) = self.color.remove(&minimising_coor).unwrap();
            assert!(self.cost[cost_at_time_of_insertion as usize].remove(&minimising_coor));
            minimising_coor
        })
    }
}

fn get_cost_squared(picture: &Picture, color: [u8; 3], r: u32, c: u32) -> f32 {
    let square = |x| {
        return x * x;
    };

    let pix_dist = |(r2, c2)| -> u32 {
        let color2 = picture.get_pix(r2, c2);

        zip(color, color2)
            .map(|(a, b)| square(a as i32 - b as i32) as u32)
            .sum()
    };

    let mut cost = 0;
    let mut n = 0;

    if c + 1 < picture.width {
        n += 1;
        cost += pix_dist((r, c + 1));
    }
    if c > 0 {
        n += 1;
        cost += pix_dist((r, c - 1));
    }
    if r > 0 {
        n += 1;
        cost += pix_dist((r - 1, c));
    }
    if r + 1 < picture.height {
        n += 1;
        cost += pix_dist((r + 1, c));
    }
    debug_assert!(n > 0, "cannot be next to all four edges");
    (cost as f32 / (3 * n) as f32)
}

fn get_cost(picture: &Picture, color: [u8; 3], r: u32, c: u32) -> u8 {
    let cost = get_cost_squared(picture, color, r, c).sqrt() as u32;

    cost.try_into()
        .unwrap_or_else(|e| panic!("{cost} Should fit in u8"))
}

async fn draw() -> anyhow::Result<()> {
    let bytes = include_bytes!("../images/tree.jpg");

    console::log_1(&format!("len = {}", bytes.len()).into());

    let img = image::io::Reader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .decode()?
        .into_rgba8();

    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| ())
        .unwrap();

    let width = img.width();
    let height = img.height();
    canvas.set_width(width);
    canvas.set_height(height);

    let overlay_canvas = document
        .get_element_by_id("canvas2")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| ())
        .unwrap();

    overlay_canvas.set_width(width);
    overlay_canvas.set_height(height);

    let context = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .unwrap();

    let overlay_context = overlay_canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .unwrap();

    let data_raw = context
        .get_image_data(0.0, 0.0, width.into(), height.into())
        .unwrap()
        .data();

    let picture = Rc::new(RefCell::new(Picture {
        data: data_raw.0.into_boxed_slice(),
        width,
        height,
    }));

    let picture2 = picture.clone();

    let metrics = Rc::new(RefCell::new(Metrics::default()));
    let metrics2 = metrics.clone();

    for (source, dest) in img
        .as_raw()
        .iter()
        .zip(picture.borrow_mut().data.iter_mut())
    {
        *dest = *source;
    }

    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();

    fn request_animation_frame(f: &Closure<dyn FnMut()>) {
        web_sys::window()
            .unwrap()
            .request_animation_frame(f.as_ref().unchecked_ref())
            .expect("should register `requestAnimationFrame` OK");
    }

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        context
            .put_image_data(
                &ImageData::new_with_u8_clamped_array(Clamped(&picture.borrow().data), width)
                    .unwrap(),
                0.0,
                0.0,
            )
            .unwrap();

        console::log_1(&format!("height {} metrics: {:?}", height, metrics).into());

        // Schedule ourself for another requestAnimationFrame callback.
        request_animation_frame(f.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));

    request_animation_frame(g.borrow().as_ref().unwrap());

    let mut el = EventLoop::new();

    loop {
        TimeoutFuture::new(0).await;

        el.tick(&mut picture2.borrow_mut(), &mut metrics2.borrow_mut());
    }
}

#[wasm_bindgen(start)]
async fn start() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    draw().await.unwrap();
}

pub struct EventLoop {
    rng: fastrand::Rng,
    candidates: Candidates,
}

impl EventLoop {
    pub fn tick(&mut self, picture: &mut Picture, metrics: &mut Metrics) {
        let swap_pix = |picture: &mut Picture, (r1, c1), (r2, c2)| {
            let n1 = r1 * picture.width + c1;
            let n2 = r2 * picture.width + c2;
            picture.data.swap(n1 as usize * 4, n2 as usize * 4);
            picture.data.swap(n1 as usize * 4 + 1, n2 as usize * 4 + 1);
            picture.data.swap(n1 as usize * 4 + 2, n2 as usize * 4 + 2);
        };

        let step = |rng: &fastrand::Rng, picture: &Picture, r, c| -> (u32, u32) {
            const NORTH: u8 = 1;
            const NORTH_EAST: u8 = 1 << 1;
            const EAST: u8 = 1 << 2;
            const SOUTH_EAST: u8 = 1 << 3;
            const SOUTH: u8 = 1 << 4;
            const SOUTH_WEST: u8 = 1 << 5;
            const WEST: u8 = 1 << 6;
            const NORTH_WEST: u8 = 1 << 7;

            let mut allowed_directions: u8 = 0xff;

            if r == 0 {
                allowed_directions &= !NORTH;
                allowed_directions &= !NORTH_EAST;
                allowed_directions &= !NORTH_WEST;
            }
            if c == 0 {
                allowed_directions &= !WEST;
                allowed_directions &= !NORTH_WEST;
                allowed_directions &= !SOUTH_WEST;
            }
            if r == picture.height - 1 {
                allowed_directions &= !SOUTH;
                allowed_directions &= !SOUTH_EAST;
                allowed_directions &= !SOUTH_WEST;
            }
            if c == picture.width - 1 {
                allowed_directions &= !EAST;
                allowed_directions &= !NORTH_EAST;
                allowed_directions &= !SOUTH_EAST;
            }

            let mut chosen_dir = rng.u8(0..(allowed_directions.count_ones() as u8));

            if (allowed_directions & NORTH) == NORTH {
                if chosen_dir == 0 {
                    return (r - 1, c);
                } else {
                    chosen_dir -= 1;
                }
            }

            if (allowed_directions & NORTH_EAST) == NORTH_EAST {
                if chosen_dir == 0 {
                    return (r - 1, c + 1);
                } else {
                    chosen_dir -= 1;
                }
            }

            if (allowed_directions & EAST) == EAST {
                if chosen_dir == 0 {
                    return (r, c + 1);
                } else {
                    chosen_dir -= 1;
                }
            }

            if (allowed_directions & SOUTH_EAST) == SOUTH_EAST {
                if chosen_dir == 0 {
                    return (r + 1, c + 1);
                } else {
                    chosen_dir -= 1;
                }
            }

            if (allowed_directions & SOUTH) == SOUTH {
                if chosen_dir == 0 {
                    return (r + 1, c);
                } else {
                    chosen_dir -= 1;
                }
            }

            if (allowed_directions & SOUTH_WEST) == SOUTH_WEST {
                if chosen_dir == 0 {
                    return (r + 1, c - 1);
                } else {
                    chosen_dir -= 1;
                }
            }

            if (allowed_directions & WEST) == WEST {
                if chosen_dir == 0 {
                    return (r, c - 1);
                } else {
                    chosen_dir -= 1;
                }
            }

            if (allowed_directions & NORTH_WEST) == NORTH_WEST {
                if chosen_dir == 0 {
                    return (r - 1, c - 1);
                } else {
                    chosen_dir -= 1;
                }
            }

            let _chosen_dir = chosen_dir;

            unreachable!();
        };


        let (mut r_b, mut c_b) = (
            self.rng.u32(0..picture.height),
            self.rng.u32(0..picture.width),
        );

        for _ in 0..10000 {
            self.candidates.pop_all_lowest_cost();

            while self.candidates.len() < 1000 {
                (r_b, c_b) = step(&self.rng, &picture, r_b, c_b);

                self.candidates.add(&picture, r_b, c_b);
            }

            let (Coordinate(r1, c1), _color1) = self
                .candidates
                .pop_highest_cost()
                .expect("coordinates full");

            let _color1 = picture.get_pix(r1, c1);
            let Coordinate(r2, c2) = self
                .candidates
                .remove_candidate_minimising(10, |candidate_color, candidate_coor| {
                    get_cost_squared(&picture, candidate_color, r1, c1)
                        + get_cost_squared(
                            &picture,
                            candidate_color,
                            candidate_coor.0,
                            candidate_coor.1,
                        )
                })
                .expect("coordinates one less than full");

            swap_pix(picture, (r1, c1), (r2, c2));

            metrics._swaps += 1;

            self.candidates.add(&picture, r2, c2);

            // overlay_context.set_fill_style(&"blue".into());
            // overlay_context.fill_rect(c1.into(), r1.into(), 1.0, 1.0);

            // overlay_context.set_fill_style(&"red".into());
            // overlay_context.fill_rect(c2.into(), r2.into(), 1.0, 1.0);
        }
    }

    pub fn new() -> Self {
        return Self {
            rng: fastrand::Rng::new(),
            candidates: Candidates::new(),
        };
    }
}
