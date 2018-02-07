use std::fmt::Debug;

pub trait Vertex: Debug + Clone {
    type Value: Debug + Clone + PartialOrd;

    fn min(&self, other: &Self) -> Self;
    fn max(&self, other: &Self) -> Self;
}

#[derive(Debug, Copy, Clone)]
pub struct Vertex3 {
    x: f64,
    y: f64,
    z: f64,
}

fn finite_min<T: PartialOrd>(a: T, b: T) -> T {
    if a < b { a } else { b }
}

fn finite_max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

impl Vertex3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        assert!(x.is_finite());
        assert!(y.is_finite());
        assert!(z.is_finite());
        Self { x: x, y: y, z: z }
    }

    pub fn get(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }
}

impl Vertex for Vertex3 {
    type Value = f64;

    fn min(&self, other: &Vertex3) -> Vertex3 {
        Vertex3 {
            x: finite_min(self.x, other.x),
            y: finite_min(self.y, other.y),
            z: finite_min(self.z, other.z),
        }
    }

    fn max(&self, other: &Vertex3) -> Vertex3 {
        Vertex3 {
            x: finite_max(self.x, other.x),
            y: finite_max(self.y, other.y),
            z: finite_max(self.z, other.z),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Rectangle<V: Vertex> {
    lo: V,
    hi: V,
}

#[derive(Debug, Clone)]
pub struct Face<V: Vertex> {
    a: V,
    b: V,
    c: V,
}

impl<V: Vertex> Face<V> {
    pub fn new(a: V, b: V, c: V) -> Self {
        Self { a: a, b: b, c: c }
    }

    pub fn bbox(&self) -> Rectangle<V> {
        Rectangle {
            lo: self.a.min(&self.b).min(&self.c),
            hi: self.a.max(&self.b).max(&self.c),
        }
    }
}

pub type Face3 = Face<Vertex3>;
pub type Rectangle3 = Rectangle<Vertex3>;

struct Interval(f64, f64);

impl Interval {
    fn disjoint(&self, other: &Interval) -> bool {
        self.1 < other.0 || other.1 < self.0
    }
}

impl Rectangle3 {
    fn xs(&self) -> Interval {
        Interval(self.lo.x, self.hi.x)
    }

    fn ys(&self) -> Interval {
        Interval(self.lo.y, self.hi.y)
    }

    pub fn disjoint_xy(&self, other: &Rectangle3) -> bool {
        self.xs().disjoint(&other.xs()) || self.ys().disjoint(&other.ys())
    }
}
