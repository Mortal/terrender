use std::os::raw::{c_double, c_ulong};
use bridge::*;
use types::*;
use predicates::order_overlapping_triangles;
use err::Result;

#[no_mangle]
pub unsafe extern "C" fn terrender_init() {
    set_panic_hook();
}

unsafe fn get_vertex(vertex: *const c_double) -> Vertex {
    Vertex(*vertex, *vertex.offset(1), *vertex.offset(2))
}

unsafe fn get_face(face: *const c_double) -> Face {
    Face(get_vertex(face), get_vertex(face.offset(3)), get_vertex(face.offset(6)))
}

export!(terrender_order_overlapping_triangles(
        faces: *const c_double, nfaces: c_ulong,
        output: *mut c_ulong, output_size: c_ulong) -> Result<c_ulong> {
    let faces = (0..nfaces).map(|i| get_face(faces.offset(9*i as isize))).collect::<Vec<_>>();
    let mut k = 0;
    let output_size = output_size as usize;
    order_overlapping_triangles(&faces, |i, j| {
        if k < output_size {
            *output.offset((2*k) as isize) = i as u64;
            *output.offset((2*k+1) as isize) = j as u64;
            k += 1;
        }
    });
    Ok(k as c_ulong)
});
