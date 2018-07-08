use types::*;

fn is_close(a: f64, b: f64) -> bool {
    let absdiff = a.max(b) - a.min(b);
    let atol = 1.0e-8;
    let rtol = 1.0e-5;
    absdiff <= atol + rtol * b.abs()
}

// Determinant. Positive iff c is left of the line from a to b.
fn orient<V: Vertex>(a: &V, b: &V, c: &V) -> f64 {
    (a.x() - c.x()) * (b.y() - c.y()) - (a.y() - c.y()) * (b.x() - c.x())
}

// Negative iff face is ordered counter-clockwise.
fn orient_face<V: Vertex>(f: &Face<V>) -> f64 {
    orient(f.a(), f.b(), f.c())
}

struct LinearBasis<T: Vertex> {
    p: T,
    q: T,
    denominator: f64,
}

impl<T: Vertex> LinearBasis<T> {
    fn onto(abscissa: T, ordinate: T) -> Self {
        let denominator = abscissa.x() * ordinate.y() - ordinate.x() * abscissa.y();
        LinearBasis { p: abscissa, q: ordinate, denominator: denominator }
    }

    fn project(&self, p: T) -> Point2 {
        Point2::new((self.q.y() * p.x() - self.q.x() * p.y()) / self.denominator,
                    (self.p.x() * p.y() - self.p.y() * p.x()) / self.denominator)
    }

    fn unproject(&self, p: Point2) -> T {
        self.p.scale(p.x()) + self.q.scale(p.y())
    }
}

struct AffineBasis<T: Vertex> {
    origin: T,
    basis: LinearBasis<T>,
}

enum Edge {
    Abscissa,
    Ordinate,
    Diagonal,
}

enum Sign {
    Inside,
    Boundary,
    Outside,
}

impl Sign {
    fn from_dist(d: f64) -> Self {
        if is_close(d, 0.0) {
            Sign::Boundary
        } else if d < 0.0 {
            Sign::Outside
        } else {
            Sign::Inside
        }
    }
}

struct FaceLocation {
    edge: Edge,
    dist: f64,
}

impl FaceLocation {
    fn from_dists(abscissa: f64, ordinate: f64, diagonal: f64) -> Self {
        FaceLocation { edge: Edge::Abscissa, dist: abscissa }
        .min(FaceLocation { edge: Edge::Ordinate, dist: ordinate })
        .min(FaceLocation { edge: Edge::Diagonal, dist: diagonal })
    }

    fn min(self, other: FaceLocation) -> Self {
        if self.dist < other.dist {
            self
        } else {
            other
        }
    }

    fn sign(&self) -> Sign {
        Sign::from_dist(self.dist)
    }
}

impl<T: Vertex> AffineBasis<T> {
    fn onto(origin: &T, abscissa: &T, ordinate: &T) -> Self {
        AffineBasis {
            origin: origin.clone(),
            basis: LinearBasis::onto(abscissa.clone() - origin.clone(),
                                     ordinate.clone() - origin.clone()),
        }
    }

    fn onto_face(face: &Face<T>) -> Self {
        Self::onto(face.a(), face.b(), face.c())
    }

    fn project(&self, p: &T) -> Point2 {
        self.basis.project(p.clone() - self.origin.clone())
    }

    fn unproject(&self, p: Point2) -> T {
        self.origin.clone() + self.basis.unproject(p)
    }

    fn locate(&self, coords: Point2) -> FaceLocation {
        FaceLocation::from_dists(
            coords.x(), coords.y(), 1.0 - (coords.x() + coords.y()))
    }
}

enum FaceOrder {
    Above,
    Below,
    Disjoint,
}

impl FaceOrder {
    fn compare(z1: f64, z2: f64) -> FaceOrder {
        if z1 < z2 { FaceOrder::Below } else { FaceOrder::Above }
    }
}

#[derive(Debug, Copy, Clone)]
struct Edge2 {
    u: Vertex3,
    v: Vertex3,
}

impl Edge2 {
    fn new(u: Vertex3, v: Vertex3) -> Self { Edge2 { u: u, v: v } }

    fn transpose(self) -> Self { Edge2::new(self.u.with_xy(self.u.y(), self.u.x()),
                                            self.v.with_xy(self.v.y(), self.v.x())) }

    fn y_intersection(self) -> Option<Vertex3> {
        let u = self.u;
        let v = self.v;
        // (u,v) must cross x=0
        if !(u.x().min(v.x()) < 0.0 && 0.0 < u.x().max(v.x())) {
            return None;
        }
        // (u,v) must not be near-vertical
        if is_close(v.x(), u.x()) {
            return None;
        }
        let dy_dx = (v.y() - u.y()) / (v.x() - u.x());
        // (u,v) must cross the vertical edge from (0,0) to (0,1)
        let y_intersection = u.y() - dy_dx * u.x();
        if !(0.0 < y_intersection && y_intersection < 1.0) {
            return None;
        }
        let dz_dx = (v.z() - u.z()) / (v.x() - u.x());
        let z = u.z() - dz_dx * u.x();
        Some(Vertex3::new(0.0, y_intersection, z))
    }

    fn x_intersection(self) -> Option<Vertex3> {
        self.transpose().y_intersection().map(|v| Vertex3::new(v.y(), v.x(), v.z()))
    }

    fn diagonal_intersection(self) -> Option<Vertex3> {
        let u = self.u;
        let v = self.v;
        // Consider the line segment on the line x+y=1
        // where -1 < x-y < 1
        let u_sum = u.x() + u.y() - 1.0;
        let u_diff = u.x() - u.y();
        let v_sum = v.x() + v.y() - 1.0;
        let v_diff = v.x() - v.y();
        if !(u_sum.min(v_sum) < 0.0 && 0.0 < u_sum.max(v_sum)) || is_close(u_sum, v_sum) {
            return None;
        }
        let dy_dx = (v_diff - u_diff) / (v_sum - u_sum);
        let sum_intersection = u_diff - dy_dx * u_sum;
        if !(-1.0 < sum_intersection && sum_intersection < 1.0) {
            return None;
        }
        let dz_dx = (v.z() - u.z()) / (v_sum - u_sum);
        let z = u.z() - dz_dx * u_sum;
        Some(Vertex3::new((sum_intersection + 1.0) / 2.0,
                          (1.0 - sum_intersection) / 2.0,
                          z))
    }
}

fn face_order_edge(f1: &Face3, p2: &Vertex3, q2: &Vertex3, r2: &Vertex3) -> FaceOrder {
    let p1 = f1.a();
    let q1 = f1.b();
    let r1 = f1.c();
    assert!(orient(p2, q2, p1) >= 0.0);
    assert!(orient(q2, r2, p1) >= 0.0);
    assert!(orient(r2, p2, p1) < 0.0);
    panic!("Not implemented")
}

fn face_order_vertex(f1: &Face3, p2: &Vertex3, q2: &Vertex3, r2: &Vertex3) -> FaceOrder {
    let p1 = f1.a();
    let q1 = f1.b();
    let r1 = f1.c();
    assert!(orient(p2, q2, p1) >= 0.0);
    assert!(orient(q2, r2, p1) < 0.0);
    assert!(orient(r2, p2, p1) < 0.0);
    panic!("Not implemented")
}

fn face_order(f1: &Face3, f2: &Face3) -> FaceOrder {
    assert!(orient_face(f1) < 0.0, "f1 not CCW: {:?}", f1);
    assert!(orient_face(f2) < 0.0, "f2 not CCW: {:?}", f2);
    if f1.bbox().disjoint_xy(&f2.bbox()) {
        return FaceOrder::Disjoint;
    }
    let p1 = f1.a();
    let q1 = f1.b();
    let r1 = f1.c();
    let p2 = f2.a();
    let q2 = f2.b();
    let r2 = f2.c();
    if orient(p2, q2, p1) >= 0.0 {
        if orient(q2, r2, p1) >= 0.0 {
            if orient(r2, p2, p1) >= 0.0 {
                panic!("p1 is inside f2")
            } else {
                // p1 in R1
                face_order_edge(f1, p2, q2, r2)
            }
        } else {
            if orient(r2, p2, p1) >= 0.0 {
                face_order_edge(f1, r2, p2, q2)
            } else {
                face_order_vertex(f1, p2, q2, r2)
            }
        }
    } else {
        if orient(q2, r2, p1) >= 0.0 {
            if orient(r2, p2, p1) >= 0.0 {
                face_order_edge(f1, q2, r2, p2)
            } else {
                face_order_vertex(f1, q2, r2, p2)
            }
        } else {
            face_order_vertex(f1, r2, p2, q2)
        }
    }

    /*
    let b1 = AffineBasis::onto_face(f1);
    let a = f2.a().with_point(b1.project(f2.a()));
    let b = f2.b().with_point(b1.project(f2.b()));
    let c = f2.c().with_point(b1.project(f2.c()));
    for e in [Edge2::new(a, b), Edge2::new(b, c), Edge2::new(c, a)].iter() {
        for option_intersection in [e.x_intersection(), e.y_intersection(), e.diagonal_intersection()].iter() {
            if let &Some(intersection) = option_intersection {
                let z1 = b1.unproject(intersection.xy()).z();
                let z2 = intersection.z();
                if !is_close(z1, z2) {
                    return FaceOrder::compare(z1, z2);
                }
            }
        }
    }
    for v in [a, b, c].iter() {
        match b1.locate(v.xy()).sign() {
            Sign::Inside =>
                return FaceOrder::compare(b1.unproject(v.xy()).z(), v.z()),
            _ => ()
        }
    }
    FaceOrder::Disjoint
    */
}

pub fn order_overlapping_triangles<F: FnMut(usize, usize) -> ()>(faces: &[Face3], mut before: F) {
    for (i1, f1) in faces.iter().enumerate() {
        for (i2, f2) in faces[0..i1].iter().enumerate() {
            match face_order(f1, f2) {
                FaceOrder::Above => before(i1, i2),
                FaceOrder::Below => before(i2, i1),
                FaceOrder::Disjoint => (),
            };
        }
    }
}
