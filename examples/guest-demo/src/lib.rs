wit_bindgen::generate!({
    world: "tvm-guest-demo",
    path: "../../wit",
    generate_all,
});

use tvm::memory::bytes;
use tvm::memory::manager;
use tvm::memory::types::RegionKind;

struct Component;

impl Guest for Component {
    fn run_test() -> u32 {
        let region = manager::create_region(RegionKind::HotHeap, 256)
            .expect("create-region");
        let h = manager::alloc(region, 4).expect("alloc");
        bytes::write(h, &[1, 2, 3, 4]).expect("write");
        let read = bytes::read(h, 4).expect("read");
        read.iter().map(|b| *b as u32).sum()
    }
}

export!(Component);
