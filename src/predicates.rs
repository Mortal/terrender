use types::*;

enum FaceOrder {
    Above,
    Below,
    Disjoint,
}

fn face_order(_f1: &Face, _f2: &Face) -> FaceOrder {
    FaceOrder::Above
}

pub fn order_overlapping_triangles<F: FnMut(usize, usize) -> ()>(faces: &[Face], mut cb: F) {
    for (i1, f1) in faces.iter().enumerate() {
        for (i2, f2) in faces[0..i1].iter().enumerate() {
            match face_order(f1, f2) {
                FaceOrder::Above => cb(i1, i2),
                FaceOrder::Below => cb(i2, i1),
                FaceOrder::Disjoint => (),
            };
        }
    }
}
