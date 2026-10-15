use std::env;

fn main() {
    // Directory watches refresh compile-time fixture lists when files are added or removed.
    println!("cargo::rerun-if-changed=tests/spec");
    println!("cargo::rerun-if-changed=tests/regressions");

    // The flat executor's handlers dispatch by tail call where the optimiser turns those calls
    // into jumps, and otherwise return to a loop, the trampoline. Tail calls are limited to the
    // architecture `scripts/check_dispatch.py` verifies. KRASM_TRAMPOLINE=1 forces the loop.
    println!("cargo::rustc-check-cfg=cfg(trampoline)");
    println!("cargo::rerun-if-env-changed=KRASM_TRAMPOLINE");
    let optimised = matches!(env::var("OPT_LEVEL").as_deref(), Ok("2" | "3" | "s" | "z"));
    let verified = env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|arch| arch == "x86_64");
    if !optimised || !verified || env::var("KRASM_TRAMPOLINE").is_ok_and(|v| v == "1") {
        println!("cargo::rustc-cfg=trampoline");
    }
}
