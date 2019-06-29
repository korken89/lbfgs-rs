/// Trait for implementing WideSlice
pub trait ToWideSlice<'a> {
    fn to_wideslice(&'a mut self) -> WideSlice<'a>;
}

/// The platform independent data storage
#[derive(Debug)]
pub struct WideSlice<'a> {
    slices: &'a mut [*mut f64],
    len: usize,
}

impl<'a> WideSlice<'a> {
    pub fn to_slice(&self, idx: usize) -> &'a [f64] {
        unsafe { std::slice::from_raw_parts(self.slices[idx], self.len) }
    }

    pub fn to_mut_slice(&mut self, idx: usize) -> &'a mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.slices[idx], self.len) }
    }
}

/// Memory that is head allocated (std)
#[derive(Debug)]
pub struct HeapMem {
    buf: Box<[f64]>,
    ptr: Box<[*mut f64]>,
}

impl HeapMem {
    pub fn new(slice_size: usize, n_slices: usize) -> HeapMem {
        let mut buf = vec![0.0; slice_size * n_slices].into_boxed_slice();
        let ptr = {
            let mut v = Vec::with_capacity(n_slices);

            for chunk in buf.chunks_exact_mut(slice_size) {
                v.push(chunk.as_mut_ptr());
            }

            v.into_boxed_slice()
        };

        HeapMem { buf: buf, ptr: ptr }
    }
}

impl<'a> ToWideSlice<'a> for HeapMem {
    fn to_wideslice(&mut self) -> WideSlice {
        let len = self.buf.len() / self.ptr.len();

        WideSlice {
            slices: &mut self.ptr,
            len: len,
        }
    }
}

/// Memory that is statically allocated (no-std)
#[derive(Debug)]
pub struct StaticMem<'a> {
    buf: &'a mut [f64],
    ptr: &'a mut [*mut f64],
}

impl<'a> StaticMem<'a> {
    pub fn new(
        slice_size: usize,
        n_slices: usize,
        buf: &'a mut [f64],
        ptr_buf: &'a mut [*mut f64],
    ) -> StaticMem<'a> {
        assert!(buf.len() >= slice_size * n_slices);
        assert!(ptr_buf.len() >= n_slices);

        StaticMem {
            buf: buf,
            ptr: ptr_buf,
        }
    }
}

impl<'a> ToWideSlice<'a> for StaticMem<'a> {
    fn to_wideslice(&'a mut self) -> WideSlice<'a> {
        let len = self.buf.len() / self.ptr.len();

        WideSlice {
            slices: &mut self.ptr,
            len: len,
        }
    }
}

/// Memory that is stack allocated (no-std)
#[derive(Debug)]
pub struct StackMem {
    // pub struct<const SLICE_SIZE: usize, const N_SLICES: usize> StackMem {
    buf: [f64; 8], // No const generics yet :(
    ptr: [*mut f64; 4],
}

impl StackMem {
    pub fn new() -> StackMem {
        StackMem {
            buf: [0.0; 8],
            ptr: [0 as *mut f64; 4],
        }
    }
}

impl<'a> ToWideSlice<'a> for StackMem {
    fn to_wideslice(&mut self) -> WideSlice {
        let len = self.buf.len() / self.ptr.len();

        for (chunk, ptr) in self.buf.chunks_exact_mut(2).zip(self.ptr.iter_mut()) {
            *ptr = chunk.as_mut_ptr()
        }

        WideSlice {
            slices: &mut self.ptr,
            len: len,
        }
    }
}

fn main() {
    let mut mem = HeapMem::new(2, 4);
    println!("mem: {:?}", mem);

    let ws = mem.to_wideslice();
    println!("ws: {:?}", ws);

    ws.slices.rotate_right(1);

    println!("ws: {:?}", ws);

    println!("s1: {:?}", ws.to_slice(0));
    println!("s2: {:?}", ws.to_slice(1));
    println!("s3: {:?}", ws.to_slice(2));
    println!("s4: {:?}", ws.to_slice(3));

    println!("");
    println!("");
    println!("");

    let mut mem = StackMem::new();
    println!("mem: {:?}", mem);

    let ws = mem.to_wideslice();
    println!("ws: {:?}", ws);

    ws.slices.rotate_right(1);

    println!("ws: {:?}", ws);

    println!("s1: {:?}", ws.to_slice(0));
    println!("s2: {:?}", ws.to_slice(1));
    println!("s3: {:?}", ws.to_slice(2));
    println!("s4: {:?}", ws.to_slice(3));
}
