#![feature(array_chunks)]

use gloo_timers::future::TimeoutFuture;

use std::cell::RefCell;

use std::collections::HashMap;

// use std::collections::btree_map::Entry;
use std::collections::hash_map::Entry;

use std::io::Cursor;
use std::iter::zip;
use std::num::NonZeroU8;
use std::num::TryFromIntError;
use std::ops;
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
    add_new: usize,
    add_old: usize,
    remove_candidate_minimising: usize,
    add_overwrites: usize,
    lowest_cost_popped: usize,
    popped_cost: f32,
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
    pub fn new(data: Box<[u8]>, width: u32, height: u32) -> Self {
        Self {
            data,
            width,
            height,
        }
    }

    fn get_data_index(&self, c: Coordinate) -> usize {
        let pix_idx = c.0 * self.width + c.1;
        (pix_idx * 4).try_into().unwrap()
    }

    fn get_pix(&self, c: Coordinate) -> [u8; 3] {
        let slice = &self.data[self.get_data_index(c)..][0..3];
        return [slice[0], slice[1], slice[2]];
    }
}

#[derive(Default, Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
struct Coordinate(u32, u32);

type CostStore = [Vec<(Coordinate, [u8; 3])>; 256];

struct CoorMap<T> {
    store: Vec<Option<T>>,
    width: u32,
}
impl<T> CoorMap<T> {
    fn new(picture: &Picture) -> CoorMap<T> {
        Self {
            store: (0..(picture.width * picture.height))
                .into_iter()
                .map(|_| None)
                .collect(),
            width: picture.width,
        }
    }
}

struct Candidates {
    /// Maps pixels coordinate to its cost at time of insertion and color. Must be kept up to date.
    color_longname: CoorMap<u8>,

    /// Maps pixels cost at time of insertion to its location
    cost_longname: CostStore,

    len: usize,
}

impl<T> ops::Index<Coordinate> for CoorMap<T> {
    type Output = Option<T>;

    fn index(&self, c: Coordinate) -> &Self::Output {
        let index = c.0 * self.width + c.1;
        return &self.store[usize::try_from(index).unwrap()];
    }
}

impl<T> ops::IndexMut<Coordinate> for CoorMap<T> {
    fn index_mut(&mut self, c: Coordinate) -> &mut Self::Output {
        let index = c.0 * self.width + c.1;
        return &mut self.store[usize::try_from(index).unwrap()];
    }
}

impl Candidates {
    fn new(picture: &Picture) -> Self {
        return Self {
            color_longname: CoorMap::new(picture),
            cost_longname: [(); 256].map(|_| Default::default()),
            len: 0,
        };
    }

    fn remove_cost(store: &mut CostStore, cost: u8, coor: &Coordinate) -> bool {
        let cost_container = &mut store[cost as usize];
        if let Some(index) = cost_container.iter().position(|(c, _)| c == coor) {
            cost_container.remove(index);
            true
        } else {
            false
        }
    }

    fn add(&mut self, picture: &Picture, coor: Coordinate) -> bool {
        let color = picture.get_pix(coor);
        let new_cost = get_cost(picture, color, coor);

        match &mut self.color_longname[coor] {
            n @ None => {
                *n = Some(new_cost);

                self.cost_longname[new_cost as usize].push((coor, color));

                self.len += 1;

                false
            }
            Some(cost_at_time_of_insertion) => {
                if !Self::remove_cost(
                    &mut self.cost_longname,
                    (*cost_at_time_of_insertion).into(),
                    &coor,
                ) {
                    self.len += 1;
                }

                self.cost_longname[new_cost as usize].push((coor, color));
                *cost_at_time_of_insertion = new_cost;

                true
            }
        }
    }

    fn pop_highest_cost(&mut self) -> Option<(u8, Coordinate)> {
        let highest_cost = {
            let mut highest_cost = 255u8;

            loop {
                if !self.cost_longname[highest_cost as usize].is_empty() {
                    break Some(highest_cost);
                }
                if highest_cost == 0 {
                    break None;
                }
                highest_cost -= 1;
            }
        };

        highest_cost.map(|highest_cost| {
            let coor = self.cost_longname[highest_cost as usize]
                .pop()
                .expect("this is not empty")
                .0;

            self.len -= 1;

            // self
            //     .color_longname
            //     .remove(&coor)
            //     .expect("must be in map as we found cost");

            (highest_cost, coor)
        })
    }

    fn pop_all_lowest_cost(&mut self) -> usize {
        let lowest_cost = {
            let mut lowest_cost = 0;

            loop {
                if !self.cost_longname[lowest_cost].is_empty() {
                    break Some(lowest_cost);
                }
                if lowest_cost == 255 {
                    break None;
                }
                lowest_cost += 1;
            }
        };

        let popped = lowest_cost
            .map(|lowest_cost| {
                let store = &mut self.cost_longname[lowest_cost];
                let count = store.len();
                store.clear();
                count
            })
            .unwrap_or(0);
        self.len -= popped;
        popped
    }

    fn remove_candidate_minimising<T: PartialOrd>(
        &mut self,
        mut check_count: usize,
        f: impl Fn([u8; 3], Coordinate) -> T,
    ) -> Option<Coordinate> {
        let mut minimising_coor = None;
        let mut minimising_cost = None;
        let mut current_min = None;

        'outer: for i in 0..=255 {
            // let mut cost = 255;
            // 'outer: loop {
            let cost = 255u8 - i;
            // if !self.cost[cost].is_empty() {
            //     dbg!(self.cost[cost].len());
            // }
            for (coor, color) in &self.cost_longname[cost as usize] {
                if check_count == 0 {
                    break 'outer;
                }
                check_count -= 1;

                let val = f(*color, *coor);
                if current_min
                    .as_ref()
                    .map_or(true, |current_min| &val <= current_min)
                {
                    current_min = Some(val);
                    minimising_cost = Some(cost);
                    minimising_coor = Some(*coor);
                }
            }
        }

        minimising_coor.map(|minimising_coor| {
            // let (cost_at_time_of_insertion,) = self.color_longname.remove(&minimising_coor).unwrap();
            assert!(Self::remove_cost(
                &mut self.cost_longname,
                minimising_cost.unwrap(),
                &minimising_coor
            ));
            self.len -= 1;
            minimising_coor
        })
    }
}

fn get_cost_squared(picture: &Picture, color: [u8; 3], c: Coordinate) -> u32 {
    // let square = |x| {
    //     return x * x;
    // };

    let pix_dist = |c2| -> u32 {
        let color2 = picture.get_pix(c2);

        (
        (color[0] as i32 - color2[0] as i32).abs() +
        (color[1] as i32 - color2[1] as i32).abs() +
        (color[2] as i32 - color2[2] as i32).abs()) as u32
    };

    let mut cost = 0;
    let mut n = 0;

    if c.1 + 1 < picture.width {
        n += 1;
        cost += pix_dist(Coordinate(c.0, c.1 + 1));
    }
    if c.1 > 0 {
        n += 1;
        cost += pix_dist(Coordinate(c.0, c.1 - 1));
    }
    if c.0 > 0 {
        n += 1;
        cost += pix_dist(Coordinate(c.0 - 1, c.1));
    }
    if c.0 + 1 < picture.height {
        n += 1;
        cost += pix_dist(Coordinate(c.0 + 1, c.1));
    }
    debug_assert!(n > 0, "cannot be next to all four edges");
    cost / (3 * n)
}

fn get_cost(picture: &Picture, color: [u8; 3], c: Coordinate) -> u8 {
    let cost = get_cost_squared(picture, color, c);

    cost.try_into()
        .unwrap_or_else(|_: TryFromIntError| panic!("{cost} Should fit in u8"))
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

    let _overlay_context = overlay_canvas
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

    let mut el = EventLoop::new(&picture2.borrow());

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
        let swap_pix = |picture: &mut Picture, c1, c2| {
            let n1 = picture.get_data_index(c1);
            let n2 = picture.get_data_index(c2);

            for i in 0..3 {
                picture.data.swap(n1 + i, n2 + i);
            }
        };

        let _step = |rng: &fastrand::Rng, picture: &Picture, r, c| -> (u32, u32) {
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

        let rand_coor = |picture: &Picture, rng: &fastrand::Rng| {
            Coordinate(rng.u32(0..picture.height), rng.u32(0..picture.width))
        };

        let candidates_len = |c: &mut Candidates| {
            if cfg!(debug_assertions) {
                let mut cost_len = 0;
                for costs in &c.cost_longname {
                    cost_len += costs.len();
                }
                assert_eq!(cost_len, c.len);
            }

            c.len
        };

        for _ in 0..10000 {
            metrics.lowest_cost_popped += self.candidates.pop_all_lowest_cost();

            while candidates_len(&mut self.candidates) < 1000 {
                let b = rand_coor(picture, &self.rng);

                // (c_b, c_b) = step(&self.rng, &picture, r_b, c_b);

                metrics.add_overwrites += self.candidates.add(&picture, b) as usize;

                metrics.add_new += 1;
            }

            let (popped_cost, c1) = self
                .candidates
                .pop_highest_cost()
                .expect("coordinates full");

            metrics.popped_cost *= 0.99;
            metrics.popped_cost += 0.01 * (popped_cost as f32);

            let c2 = /* if self.rng.u32(0..1000) == 0 {
                rand_coor(picture, &self.rng)
            } else */ {
                let color1 = picture.get_pix(c1);
                metrics.remove_candidate_minimising += 1;
                self.candidates
                    .remove_candidate_minimising(10, |candidate_color, candidate_coor| {
                        get_cost_squared(&picture, candidate_color, c1)
                            + get_cost_squared(&picture, color1, candidate_coor)
                    })
                    .expect("coordinates one less than full")
            };

            swap_pix(picture, c1, c2);

            metrics._swaps += 1;

            metrics.add_overwrites += self.candidates.add(&picture, c2) as usize;

            metrics.add_old += 1;

            // overlay_context.set_fill_style(&"blue".into());
            // overlay_context.fill_rect(c1.into(), r1.into(), 1.0, 1.0);

            // overlay_context.set_fill_style(&"red".into());
            // overlay_context.fill_rect(c2.into(), r2.into(), 1.0, 1.0);
        }
    }

    pub fn new(picture: &Picture) -> Self {
        return Self {
            rng: fastrand::Rng::new(),
            candidates: Candidates::new(picture),
        };
    }
}
