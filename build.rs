fn main() {
    // Directory watches refresh compile-time fixture lists when files are added or removed.
    println!("cargo:rerun-if-changed=tests/spec");
    println!("cargo:rerun-if-changed=tests/regressions");
}
