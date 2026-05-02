#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    Hot,
    Warm,
    Cold,
    External,
}
