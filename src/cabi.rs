use std::os::raw::{c_double, c_ulong};
use bridge::*;
use types::*;
use predicates::order_overlapping_triangles;
use err::Result;

#[no_mangle]
pub unsafe extern "C" fn terrender_init() {
    set_panic_hook();
}

unsafe fn get_vertex(vertex: *const c_double) -> Vertex3 {
    Vertex3::new(*vertex, *vertex.offset(1), *vertex.offset(2))
}

unsafe fn get_face(face: *const c_double) -> Face3 {
    Face3::new(get_vertex(face), get_vertex(face.offset(3)), get_vertex(face.offset(6)))
}

unsafe fn get_faces(faces: *const c_double, nfaces: c_ulong) -> Vec<Face3> {
    (0..nfaces).map(|i| get_face(faces.offset(9*i as isize))).collect::<Vec<_>>()
}

export!(terrender_order_overlapping_triangles(
        faces: *const c_double, nfaces: c_ulong,
        output: *mut c_ulong, output_size: c_ulong) -> Result<c_ulong> {
    let faces = get_faces(faces, nfaces);
    let mut k = 0;
    let output_size = output_size as usize;
    order_overlapping_triangles(&faces, |i, j| {
        if k < output_size {
            *output.offset((2*k) as isize) = i as c_ulong;
            *output.offset((2*k+1) as isize) = j as c_ulong;
            k += 1;
        }
    });
    Ok(k as c_ulong)
});
