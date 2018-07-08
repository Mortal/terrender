use types::*;

fn is_close(a: f64, b: f64) -> bool {
    let absdiff = a.max(b) - a.min(b);
    let atol = 1.0e-8;
    let rtol = 1.0e-5;
    absdiff <= atol + rtol * b.abs()
}

// Determinant. Positive iff c is left of the line from a to b.
fn determinant<V: Vertex>(a: &V, b: &V, c: &V) -> f64 {
    (a.x() - c.x()) * (b.y() - c.y()) - (a.y() - c.y()) * (b.x() - c.x())
}

// Negative iff face is ordered counter-clockwise.
fn orient_face<V: Vertex>(f: &Face<V>) -> f64 {
    determinant(f.a(), f.b(), f.c())
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
    assert!(determinant(p2, q2, p1) >= 0.0);
    assert!(determinant(q2, r2, p1) >= 0.0);
    assert!(determinant(r2, p2, p1) < 0.0);
    panic!("Not implemented")
}

fn face_order_vertex(f1: &Face3, p2: &Vertex3, q2: &Vertex3, r2: &Vertex3) -> FaceOrder {
    let p1 = f1.a();
    let q1 = f1.b();
    let r1 = f1.c();
    assert!(determinant(p2, q2, p1) >= 0.0);
    assert!(determinant(q2, r2, p1) < 0.0);
    assert!(determinant(r2, p2, p1) < 0.0);
    panic!("Not implemented")
}

#[derive(PartialEq)]
enum TriangleCorner {
    A, B, C
}

enum FaceOrientation {
    // x is inside
    Inside,
    // x is on corner
    OnVertex(TriangleCorner),
    // x is on opposite edge
    OnEdge(TriangleCorner),
    // x is left of opposite edge, but not adjacents
    OutsideVertex(TriangleCorner),
    // x is left of adjacent edges, but not opposite
    OutsideEdge(TriangleCorner),
}

impl FaceOrientation {
    fn on_a(f: &Face3) -> Self { FaceOrientation::OnVertex(TriangleCorner::A) }
    fn on_b(f: &Face3) -> Self { FaceOrientation::OnVertex(TriangleCorner::B) }
    fn on_c(f: &Face3) -> Self { FaceOrientation::OnVertex(TriangleCorner::C) }
    fn on_ab(f: &Face3) -> Self { FaceOrientation::OnEdge(TriangleCorner::C) }
    fn on_bc(f: &Face3) -> Self { FaceOrientation::OnEdge(TriangleCorner::A) }
    fn on_ca(f: &Face3) -> Self { FaceOrientation::OnEdge(TriangleCorner::B) }
    fn left_of_ab(f: &Face3) -> Self { FaceOrientation::OutsideVertex(TriangleCorner::C) }
    fn left_of_bc(f: &Face3) -> Self { FaceOrientation::OutsideVertex(TriangleCorner::A) }
    fn left_of_ca(f: &Face3) -> Self { FaceOrientation::OutsideVertex(TriangleCorner::B) }
    fn not_left_of_ab(f: &Face3) -> Self { FaceOrientation::OutsideEdge(TriangleCorner::C) }
    fn not_left_of_bc(f: &Face3) -> Self { FaceOrientation::OutsideEdge(TriangleCorner::A) }
    fn not_left_of_ca(f: &Face3) -> Self { FaceOrientation::OutsideEdge(TriangleCorner::B) }
}

fn on_left(p: &Vertex3, q: &Vertex3, r: &Vertex3) -> Sign {
    Sign::from_dist(determinant(p, q, r))
}

fn face_orient(face: &Face3, x: &Vertex3) -> Result<FaceOrientation, String> {
    match (on_left(face.a(), face.b(), x),
           on_left(face.b(), face.c(), x),
           on_left(face.c(), face.a(), x)) {
        (Sign::Inside, Sign::Inside, Sign::Inside) =>
            Ok(FaceOrientation::Inside),
        (Sign::Boundary, Sign::Inside, Sign::Boundary) =>
            Ok(FaceOrientation::on_a(face)),
        (Sign::Boundary, Sign::Boundary, Sign::Inside) =>
            Ok(FaceOrientation::on_b(face)),
        (Sign::Inside, Sign::Boundary, Sign::Boundary) =>
            Ok(FaceOrientation::on_c(face)),
        (Sign::Boundary, Sign::Inside, Sign::Inside) =>
            Ok(FaceOrientation::on_ab(face)),
        (Sign::Inside, Sign::Boundary, Sign::Inside) =>
            Ok(FaceOrientation::on_bc(face)),
        (Sign::Inside, Sign::Inside, Sign::Boundary) =>
            Ok(FaceOrientation::on_ca(face)),
        (Sign::Inside, Sign::Boundary, Sign::Outside) =>
            Ok(FaceOrientation::left_of_ab(face)),
        (Sign::Inside, Sign::Outside, Sign::Boundary) =>
            Ok(FaceOrientation::left_of_ab(face)),
        (Sign::Inside, Sign::Outside, Sign::Outside) =>
            Ok(FaceOrientation::left_of_ab(face)),
        (Sign::Boundary, Sign::Inside, Sign::Outside) =>
            Ok(FaceOrientation::left_of_bc(face)),
        (Sign::Outside, Sign::Inside, Sign::Boundary) =>
            Ok(FaceOrientation::left_of_bc(face)),
        (Sign::Outside, Sign::Inside, Sign::Outside) =>
            Ok(FaceOrientation::left_of_bc(face)),
        (Sign::Boundary, Sign::Outside, Sign::Inside) =>
            Ok(FaceOrientation::left_of_ca(face)),
        (Sign::Outside, Sign::Boundary, Sign::Inside) =>
            Ok(FaceOrientation::left_of_ca(face)),
        (Sign::Outside, Sign::Outside, Sign::Inside) =>
            Ok(FaceOrientation::left_of_ca(face)),
        (Sign::Outside, Sign::Inside, Sign::Inside) =>
            Ok(FaceOrientation::not_left_of_ab(face)),
        (Sign::Inside, Sign::Outside, Sign::Inside) =>
            Ok(FaceOrientation::not_left_of_bc(face)),
        (Sign::Inside, Sign::Inside, Sign::Outside) =>
            Ok(FaceOrientation::not_left_of_ca(face)),
        (Sign::Boundary, Sign::Boundary, Sign::Boundary) =>
            Err(format!("face is flat: {:?}", face)),
        (Sign::Boundary, Sign::Boundary, Sign::Outside) =>
            Err(format!("face is clockwise: {:?}", face)),
        (Sign::Boundary, Sign::Outside, Sign::Boundary) =>
            Err(format!("face is clockwise: {:?}", face)),
        (Sign::Boundary, Sign::Outside, Sign::Outside) =>
            Err(format!("face is clockwise: {:?}", face)),
        (Sign::Outside, Sign::Boundary, Sign::Boundary) =>
            Err(format!("face is clockwise: {:?}", face)),
        (Sign::Outside, Sign::Boundary, Sign::Outside) =>
            Err(format!("face is clockwise: {:?}", face)),
        (Sign::Outside, Sign::Outside, Sign::Boundary) =>
            Err(format!("face is clockwise: {:?}", face)),
        (Sign::Outside, Sign::Outside, Sign::Outside) =>
            Err(format!("face is clockwise: {:?}", face)),
    }
}

// f2 has a vertex on c1 and a vertex on c2
fn overlap_point_incident_edge(f1: &Face3, f2: &Face3, c1: TriangleCorner, c2: TriangleCorner, o3: FaceOrientation) -> Option<Vertex3> {
    assert!(c1 != c2);
    match o3 {
        FaceOrientation::Inside => panic!("Inside should be handled in overlap_point"),
        FaceOrientation::OnEdge(_) => panic!("OnEdge should be handled in overlap_point"),
        FaceOrientation::OnVertex(c3) => {
            assert!(c1 != c3);
            assert!(c2 != c3);
            Some(f2.midpoint())
        },
        FaceOrientation::OutsideVertex(c3) => {
            if c1 != c3 && c2 != c3 {
                Some(f1.midpoint())
            } else {
                None
            }
        },
        FaceOrientation::OutsideEdge(c3) => {
            if c1 != c3 && c2 != c3 {
                None
            } else {
                Some(f1.midpoint())
            }
        },
    }
}

// f2 has a vertex on c1
fn overlap_point_incident_vertex(f1: &Face3, f2: &Face3, c1: TriangleCorner, o2: FaceOrientation, o3: FaceOrientation) -> Option<Vertex3> {
    match (o2, o3) {
        (FaceOrientation::OutsideVertex(c2),
         FaceOrientation::OutsideVertex(c3)) =>
            if c1 != c2 && c2 != c3 && c3 != c1 {
                Some(f1.midpoint())
            } else {
                None
            }
        (FaceOrientation::OutsideVertex(c2),
         FaceOrientation::OutsideEdge(c3)) =>
            panic!("TODO"),
        (FaceOrientation::OutsideEdge(c2),
         FaceOrientation::OutsideVertex(c3)) =>
            panic!("TODO"),
        (FaceOrientation::OutsideEdge(c2),
         FaceOrientation::OutsideEdge(c3)) =>
            panic!("TODO"),
        _ => panic!("Remaining cases should be handled by overlap_point"),
    }
}

fn overlap_point(f1: &Face3, f2: &Face3) -> Option<Vertex3> {
    match (face_orient(f1, f2.a()).unwrap(),
           face_orient(f1, f2.b()).unwrap(),
           face_orient(f1, f2.c()).unwrap()) {
        (FaceOrientation::Inside, _, _) =>
            // f2.a() is a proper overlap point
            Some(f2.a().clone()),
        (FaceOrientation::OnEdge(_), _, _) =>
            // f2.a() is not really a proper overlap point,
            // but in terrender, a 3D vertex is never contained in a 3D edge,
            // so we can determine z-order by using f2.a() as overlap point.
            // TODO: Find proper overlap point in this case.
            Some(f2.a().clone()),
        (_, FaceOrientation::Inside, _) =>
            Some(f2.b().clone()),
        (_, FaceOrientation::OnEdge(_), _) =>
            Some(f2.b().clone()),
        (_, _, FaceOrientation::Inside) =>
            Some(f2.c().clone()),
        (_, _, FaceOrientation::OnEdge(_)) =>
            Some(f2.c().clone()),
        (FaceOrientation::OnVertex(c1),
         FaceOrientation::OnVertex(c2),
         o3) =>
            overlap_point_incident_edge(f1, f2, c1, c2, o3),
        (FaceOrientation::OnVertex(c1),
         o2,
         FaceOrientation::OnVertex(c3)) =>
            overlap_point_incident_edge(f1, f2, c3, c1, o2),
        (o1,
         FaceOrientation::OnVertex(c2),
         FaceOrientation::OnVertex(c3)) =>
            overlap_point_incident_edge(f1, f2, c2, c3, o1),
        (FaceOrientation::OnVertex(c1), o2, o3) =>
            overlap_point_incident_vertex(f1, f2, c1, o2, o3),
        (o1, FaceOrientation::OnVertex(c2), o3) =>
            overlap_point_incident_vertex(f1, f2, c2, o3, o1),
        (o1, o2, FaceOrientation::OnVertex(c3)) =>
            overlap_point_incident_vertex(f1, f2, c3, o1, o2),
        _ => panic!("TODO"),
        // FaceOrientation::OnVertex(c1) => panic!("TODO"),
        // FaceOrientation::OutsideVertex(c1) => panic!("TODO"),
        // FaceOrientation::OutsideEdge(c1) => panic!("TODO"),
    }
}

fn face_order(f1: &Face3, f2: &Face3) -> FaceOrder {
    if f1.bbox().disjoint_xy(&f2.bbox()) {
        return FaceOrder::Disjoint;
    }
    if let Some(p) = overlap_point(f1, f2) {
        let b1 = AffineBasis::onto_face(f1);
        let z1 = b1.unproject(p.xy()).z();
        let z2 = p.z();
        return FaceOrder::compare(z1, z2);
    }
    FaceOrder::Disjoint
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
