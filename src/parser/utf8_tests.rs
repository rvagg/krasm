// AUTO-GENERATED FILE - DO NOT EDIT
// Generated from WebAssembly spec UTF-8 tests

#[cfg(test)]
mod tests {
    use crate::parser;

    #[derive(Debug)]
    struct Utf8TestCase {
        name: &'static str,
        binary: &'static [u8],
        expected_error: &'static str,
    }

    const UTF8_TEST_CASES: &[Utf8TestCase] = &[
        Utf8TestCase {
            name: "utf8-custom-section-id:7",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 2, 1, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:17",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 2, 1, 143],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:27",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 2, 1, 144],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:37",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 2, 1, 159],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:47",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 2, 1, 160],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:57",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 2, 1, 191],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:69",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 194, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:79",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 2, 1, 194],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:89",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 194, 46],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:101",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 192, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:111",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 192, 191],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:121",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 193, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:131",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 193, 191],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:141",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 194, 0],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:151",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 194, 127],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:161",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 194, 192],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:171",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 194, 253],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:181",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 223, 0],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:191",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 223, 127],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:201",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 223, 192],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:211",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 223, 253],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:223",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 225, 128, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:233",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 225, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:243",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 225, 128, 46],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:253",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 2, 1, 225],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:263",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 225, 46],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:275",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 224, 0, 160],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:285",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 224, 127, 160],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:295",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 224, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:305",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 224, 128, 160],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:315",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 224, 159, 160],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:325",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 224, 159, 191],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:335",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 224, 192, 160],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:345",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 224, 253, 160],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:355",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 225, 0, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:365",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 225, 127, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:375",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 225, 192, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:385",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 225, 253, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:395",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 236, 0, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:405",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 236, 127, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:415",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 236, 192, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:425",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 236, 253, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:435",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 237, 0, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:445",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 237, 127, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:455",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 237, 160, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:465",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 237, 160, 191],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:475",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 237, 191, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:485",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 237, 191, 191],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:495",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 237, 192, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:505",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 237, 253, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:515",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 238, 0, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:525",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 238, 127, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:535",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 238, 192, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:545",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 238, 253, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:555",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 239, 0, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:565",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 239, 127, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:575",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 239, 192, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:585",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 239, 253, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:597",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 224, 160, 0],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:607",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 224, 160, 127],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:617",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 224, 160, 192],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:627",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 224, 160, 253],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:637",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 225, 128, 0],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:647",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 225, 128, 127],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:657",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 225, 128, 192],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:667",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 225, 128, 253],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:677",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 236, 128, 0],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:687",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 236, 128, 127],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:697",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 236, 128, 192],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:707",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 236, 128, 253],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:717",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 237, 128, 0],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:727",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 237, 128, 127],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:737",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 237, 128, 192],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:747",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 237, 128, 253],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:757",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 238, 128, 0],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:767",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 238, 128, 127],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:777",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 238, 128, 192],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:787",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 238, 128, 253],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:797",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 239, 128, 0],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:807",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 239, 128, 127],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:817",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 239, 128, 192],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:827",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 239, 128, 253],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:839",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 6, 5, 241, 128, 128, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:849",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 241, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:859",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 241, 128, 128, 35],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:869",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 241, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:879",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 241, 128, 35],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:889",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 2, 1, 241],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:899",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 241, 35],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:911",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 240, 0, 144, 144],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:921",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 240, 127, 144, 144],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:931",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 240, 128, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:941",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 240, 128, 144, 144],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:951",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 240, 143, 144, 144],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:961",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 240, 143, 191, 191],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:971",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 240, 192, 144, 144],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:981",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 240, 253, 144, 144],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:991",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 241, 0, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1001",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 241, 127, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1011",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 241, 192, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1021",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 241, 253, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1031",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 243, 0, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1041",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 243, 127, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1051",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 243, 192, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1061",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 243, 253, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1071",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 244, 0, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1081",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 244, 127, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1091",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 244, 144, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1101",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 244, 191, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1111",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 244, 192, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1121",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 244, 253, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1131",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 245, 128, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1141",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 247, 128, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1151",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 247, 191, 191, 191],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1163",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 240, 144, 0, 144],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1173",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 240, 144, 127, 144],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1183",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 240, 144, 192, 144],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1193",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 240, 144, 253, 144],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1203",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 241, 128, 0, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1213",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 241, 128, 127, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1223",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 241, 128, 192, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1233",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 241, 128, 253, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1243",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 243, 128, 0, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1253",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 243, 128, 127, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1263",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 243, 128, 192, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1273",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 243, 128, 253, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1283",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 244, 128, 0, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1293",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 244, 128, 127, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1303",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 244, 128, 192, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1313",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 244, 128, 253, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1325",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 240, 144, 144, 0],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1335",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 240, 144, 144, 127],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1345",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 240, 144, 144, 192],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1355",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 240, 144, 144, 253],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1365",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 241, 128, 128, 0],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1375",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 241, 128, 128, 127],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1385",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 241, 128, 128, 192],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1395",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 241, 128, 128, 253],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1405",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 243, 128, 128, 0],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1415",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 243, 128, 128, 127],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1425",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 243, 128, 128, 192],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1435",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 243, 128, 128, 253],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1445",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 244, 128, 128, 0],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1455",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 244, 128, 128, 127],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1465",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 244, 128, 128, 192],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1475",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 244, 128, 128, 253],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1487",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 7, 6, 248, 128, 128, 128, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1497",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 248, 128, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1507",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 6, 5, 248, 128, 128, 128, 35],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1517",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 248, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1527",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 248, 128, 128, 35],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1537",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 248, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1547",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 248, 128, 35],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1557",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 2, 1, 248],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1567",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 248, 35],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1579",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 6, 5, 248, 128, 128, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1589",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 6, 5, 251, 191, 191, 191, 191],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1601",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 8, 7, 252, 128, 128, 128, 128, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1611",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 6, 5, 252, 128, 128, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1621",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 7, 6, 252, 128, 128, 128, 128, 35],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1631",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 252, 128, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1641",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 6, 5, 252, 128, 128, 128, 35],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1651",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 252, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1661",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 252, 128, 128, 35],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1671",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 252, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1681",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 4, 3, 252, 128, 35],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1691",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 2, 1, 252],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1701",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 252, 35],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1713",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 7, 6, 252, 128, 128, 128, 128, 128],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1723",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 7, 6, 253, 191, 191, 191, 191, 191],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1735",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 2, 1, 254],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1745",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 2, 1, 255],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1755",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 254, 255],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1765",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 0, 0, 254, 255],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1775",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 3, 2, 255, 254],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-custom-section-id:1785",
            binary: &[0, 97, 115, 109, 1, 0, 0, 0, 0, 5, 4, 255, 254, 0, 0],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:7",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 1, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:22",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 1, 143, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:37",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 1, 144, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:52",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 1, 159, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:67",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 1, 160, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:82",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 1, 191, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:99",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 194, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:114",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 1, 194, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:129",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 194, 46, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:146",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 192, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:161",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 192, 191, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:176",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 193, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:191",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 193, 191, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:206",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 194, 0, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:221",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 194, 127, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:236",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 194, 192, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:251",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 194, 253, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:266",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 223, 0, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:281",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 223, 127, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:296",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 223, 192, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:311",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 223, 253, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:328",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 225, 128, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:343",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 225, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:358",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 225, 128, 46, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:373",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 1, 225, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:388",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 225, 46, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:405",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 224, 0, 160, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:420",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 224, 127, 160, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:435",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 224, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:450",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 224, 128, 160, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:465",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 224, 159, 160, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:480",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 224, 159, 191, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:495",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 224, 192, 160, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:510",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 224, 253, 160, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:525",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 225, 0, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:540",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 225, 127, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:555",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 225, 192, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:570",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 225, 253, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:585",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 236, 0, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:600",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 236, 127, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:615",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 236, 192, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:630",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 236, 253, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:645",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 237, 0, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:660",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 237, 127, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:675",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 237, 160, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:690",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 237, 160, 191, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:705",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 237, 191, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:720",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 237, 191, 191, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:735",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 237, 192, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:750",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 237, 253, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:765",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 238, 0, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:780",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 238, 127, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:795",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 238, 192, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:810",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 238, 253, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:825",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 239, 0, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:840",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 239, 127, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:855",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 239, 192, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:870",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 239, 253, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:887",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 224, 160, 0, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:902",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 224, 160, 127, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:917",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 224, 160, 192, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:932",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 224, 160, 253, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:947",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 225, 128, 0, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:962",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 225, 128, 127, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:977",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 225, 128, 192, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:992",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 225, 128, 253, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1007",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 236, 128, 0, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1022",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 236, 128, 127, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1037",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 236, 128, 192, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1052",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 236, 128, 253, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1067",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 237, 128, 0, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1082",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 237, 128, 127, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1097",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 237, 128, 192, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1112",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 237, 128, 253, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1127",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 238, 128, 0, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1142",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 238, 128, 127, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1157",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 238, 128, 192, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1172",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 238, 128, 253, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1187",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 239, 128, 0, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1202",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 239, 128, 127, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1217",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 239, 128, 192, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1232",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 239, 128, 253, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1249",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 15, 1, 5, 241, 128, 128, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1264",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 241, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1279",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 241, 128, 128, 35, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1294",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 241, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1309",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 241, 128, 35, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1324",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 1, 241, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1339",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 241, 35, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1356",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 240, 0, 144, 144, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1371",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 240, 127, 144, 144, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1386",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 240, 128, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1401",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 240, 128, 144, 144, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1416",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 240, 143, 144, 144, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1431",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 240, 143, 191, 191, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1446",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 240, 192, 144, 144, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1461",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 240, 253, 144, 144, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1476",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 241, 0, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1491",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 241, 127, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1506",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 241, 192, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1521",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 241, 253, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1536",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 243, 0, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1551",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 243, 127, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1566",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 243, 192, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1581",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 243, 253, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1596",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 244, 0, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1611",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 244, 127, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1626",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 244, 144, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1641",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 244, 191, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1656",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 244, 192, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1671",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 244, 253, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1686",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 245, 128, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1701",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 247, 128, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1716",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 247, 191, 191, 191, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1733",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 240, 144, 0, 144, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1748",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 240, 144, 127, 144, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1763",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 240, 144, 192, 144, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1778",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 240, 144, 253, 144, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1793",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 241, 128, 0, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1808",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 241, 128, 127, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1823",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 241, 128, 192, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1838",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 241, 128, 253, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1853",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 243, 128, 0, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1868",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 243, 128, 127, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1883",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 243, 128, 192, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1898",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 243, 128, 253, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1913",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 244, 128, 0, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1928",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 244, 128, 127, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1943",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 244, 128, 192, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1958",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 244, 128, 253, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1975",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 240, 144, 144, 0, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:1990",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 240, 144, 144, 127, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2005",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 240, 144, 144, 192, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2020",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 240, 144, 144, 253, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2035",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 241, 128, 128, 0, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2050",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 241, 128, 128, 127, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2065",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 241, 128, 128, 192, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2080",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 241, 128, 128, 253, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2095",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 243, 128, 128, 0, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2110",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 243, 128, 128, 127, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2125",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 243, 128, 128, 192, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2140",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 243, 128, 128, 253, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2155",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 244, 128, 128, 0, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2170",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 244, 128, 128, 127, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2185",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 244, 128, 128, 192, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2200",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 244, 128, 128, 253, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2217",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 16, 1, 6, 248, 128, 128, 128, 128, 128, 4, 116, 101, 115, 116, 3, 127,
                0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2232",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 248, 128, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2247",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 15, 1, 5, 248, 128, 128, 128, 35, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2262",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 248, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2277",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 248, 128, 128, 35, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2292",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 248, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2307",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 248, 128, 35, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2322",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 1, 248, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2337",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 248, 35, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2354",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 15, 1, 5, 248, 128, 128, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2369",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 15, 1, 5, 251, 191, 191, 191, 191, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2386",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 17, 1, 7, 252, 128, 128, 128, 128, 128, 128, 4, 116, 101, 115, 116, 3,
                127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2401",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 15, 1, 5, 252, 128, 128, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2416",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 16, 1, 6, 252, 128, 128, 128, 128, 35, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2431",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 252, 128, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2446",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 15, 1, 5, 252, 128, 128, 128, 35, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2461",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 252, 128, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2476",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 252, 128, 128, 35, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2491",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 252, 128, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2506",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 3, 252, 128, 35, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2521",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 1, 252, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2536",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 252, 35, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2553",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 16, 1, 6, 252, 128, 128, 128, 128, 128, 4, 116, 101, 115, 116, 3, 127,
                0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2568",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 16, 1, 6, 253, 191, 191, 191, 191, 191, 4, 116, 101, 115, 116, 3, 127,
                0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2585",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 1, 254, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2600",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 1, 255, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2615",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 254, 255, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2630",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 0, 0, 254, 255, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2645",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 2, 255, 254, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-field:2660",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 255, 254, 0, 0, 4, 116, 101, 115, 116, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:7",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 4, 116, 101, 115, 116, 1, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:22",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 4, 116, 101, 115, 116, 1, 143, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:37",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 4, 116, 101, 115, 116, 1, 144, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:52",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 4, 116, 101, 115, 116, 1, 159, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:67",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 4, 116, 101, 115, 116, 1, 160, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:82",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 4, 116, 101, 115, 116, 1, 191, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:99",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 194, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:114",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 4, 116, 101, 115, 116, 1, 194, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:129",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 194, 46, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:146",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 192, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:161",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 192, 191, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:176",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 193, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:191",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 193, 191, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:206",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 194, 0, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:221",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 194, 127, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:236",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 194, 192, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:251",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 194, 253, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:266",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 223, 0, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:281",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 223, 127, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:296",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 223, 192, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:311",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 223, 253, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:328",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 225, 128, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:343",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 225, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:358",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 225, 128, 46, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:373",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 4, 116, 101, 115, 116, 1, 225, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:388",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 225, 46, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:405",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 224, 0, 160, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:420",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 224, 127, 160, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:435",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 224, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:450",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 224, 128, 160, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:465",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 224, 159, 160, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:480",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 224, 159, 191, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:495",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 224, 192, 160, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:510",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 224, 253, 160, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:525",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 225, 0, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:540",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 225, 127, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:555",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 225, 192, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:570",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 225, 253, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:585",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 236, 0, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:600",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 236, 127, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:615",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 236, 192, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:630",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 236, 253, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:645",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 237, 0, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:660",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 237, 127, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:675",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 237, 160, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:690",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 237, 160, 191, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:705",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 237, 191, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:720",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 237, 191, 191, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:735",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 237, 192, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:750",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 237, 253, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:765",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 238, 0, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:780",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 238, 127, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:795",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 238, 192, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:810",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 238, 253, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:825",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 239, 0, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:840",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 239, 127, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:855",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 239, 192, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:870",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 239, 253, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:887",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 224, 160, 0, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:902",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 224, 160, 127, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:917",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 224, 160, 192, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:932",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 224, 160, 253, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:947",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 225, 128, 0, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:962",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 225, 128, 127, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:977",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 225, 128, 192, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:992",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 225, 128, 253, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1007",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 236, 128, 0, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1022",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 236, 128, 127, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1037",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 236, 128, 192, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1052",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 236, 128, 253, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1067",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 237, 128, 0, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1082",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 237, 128, 127, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1097",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 237, 128, 192, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1112",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 237, 128, 253, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1127",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 238, 128, 0, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1142",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 238, 128, 127, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1157",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 238, 128, 192, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1172",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 238, 128, 253, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1187",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 239, 128, 0, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1202",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 239, 128, 127, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1217",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 239, 128, 192, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1232",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 239, 128, 253, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1249",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 15, 1, 4, 116, 101, 115, 116, 5, 241, 128, 128, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1264",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 241, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1279",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 241, 128, 128, 35, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1294",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 241, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1309",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 241, 128, 35, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1324",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 4, 116, 101, 115, 116, 1, 241, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1339",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 241, 35, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1356",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 240, 0, 144, 144, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1371",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 240, 127, 144, 144, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1386",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 240, 128, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1401",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 240, 128, 144, 144, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1416",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 240, 143, 144, 144, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1431",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 240, 143, 191, 191, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1446",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 240, 192, 144, 144, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1461",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 240, 253, 144, 144, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1476",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 241, 0, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1491",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 241, 127, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1506",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 241, 192, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1521",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 241, 253, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1536",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 243, 0, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1551",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 243, 127, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1566",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 243, 192, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1581",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 243, 253, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1596",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 244, 0, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1611",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 244, 127, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1626",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 244, 144, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1641",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 244, 191, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1656",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 244, 192, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1671",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 244, 253, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1686",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 245, 128, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1701",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 247, 128, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1716",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 247, 191, 191, 191, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1733",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 240, 144, 0, 144, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1748",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 240, 144, 127, 144, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1763",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 240, 144, 192, 144, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1778",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 240, 144, 253, 144, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1793",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 241, 128, 0, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1808",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 241, 128, 127, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1823",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 241, 128, 192, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1838",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 241, 128, 253, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1853",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 243, 128, 0, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1868",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 243, 128, 127, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1883",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 243, 128, 192, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1898",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 243, 128, 253, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1913",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 244, 128, 0, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1928",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 244, 128, 127, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1943",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 244, 128, 192, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1958",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 244, 128, 253, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1975",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 240, 144, 144, 0, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:1990",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 240, 144, 144, 127, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2005",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 240, 144, 144, 192, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2020",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 240, 144, 144, 253, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2035",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 241, 128, 128, 0, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2050",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 241, 128, 128, 127, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2065",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 241, 128, 128, 192, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2080",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 241, 128, 128, 253, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2095",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 243, 128, 128, 0, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2110",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 243, 128, 128, 127, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2125",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 243, 128, 128, 192, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2140",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 243, 128, 128, 253, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2155",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 244, 128, 128, 0, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2170",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 244, 128, 128, 127, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2185",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 244, 128, 128, 192, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2200",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 244, 128, 128, 253, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2217",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 16, 1, 4, 116, 101, 115, 116, 6, 248, 128, 128, 128, 128, 128, 3, 127,
                0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2232",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 248, 128, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2247",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 15, 1, 4, 116, 101, 115, 116, 5, 248, 128, 128, 128, 35, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2262",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 248, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2277",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 248, 128, 128, 35, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2292",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 248, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2307",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 248, 128, 35, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2322",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 4, 116, 101, 115, 116, 1, 248, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2337",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 248, 35, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2354",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 15, 1, 4, 116, 101, 115, 116, 5, 248, 128, 128, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2369",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 15, 1, 4, 116, 101, 115, 116, 5, 251, 191, 191, 191, 191, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2386",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 17, 1, 4, 116, 101, 115, 116, 7, 252, 128, 128, 128, 128, 128, 128, 3,
                127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2401",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 15, 1, 4, 116, 101, 115, 116, 5, 252, 128, 128, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2416",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 16, 1, 4, 116, 101, 115, 116, 6, 252, 128, 128, 128, 128, 35, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2431",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 252, 128, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2446",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 15, 1, 4, 116, 101, 115, 116, 5, 252, 128, 128, 128, 35, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2461",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 252, 128, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2476",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 252, 128, 128, 35, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2491",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 252, 128, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2506",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 13, 1, 4, 116, 101, 115, 116, 3, 252, 128, 35, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2521",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 4, 116, 101, 115, 116, 1, 252, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2536",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 252, 35, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2553",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 16, 1, 4, 116, 101, 115, 116, 6, 252, 128, 128, 128, 128, 128, 3, 127,
                0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2568",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 16, 1, 4, 116, 101, 115, 116, 6, 253, 191, 191, 191, 191, 191, 3, 127,
                0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2585",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 4, 116, 101, 115, 116, 1, 254, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2600",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 11, 1, 4, 116, 101, 115, 116, 1, 255, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2615",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 254, 255, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2630",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 0, 0, 254, 255, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2645",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 12, 1, 4, 116, 101, 115, 116, 2, 255, 254, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
        Utf8TestCase {
            name: "utf8-import-module:2660",
            binary: &[
                0, 97, 115, 109, 1, 0, 0, 0, 2, 14, 1, 4, 116, 101, 115, 116, 4, 255, 254, 0, 0, 3, 127, 0,
            ],
            expected_error: "malformed UTF-8 encoding",
        },
    ];

    #[test]
    fn test_utf8_validation() {
        let mut passed = 0;
        let mut failed = 0;
        let module_registry = std::collections::HashMap::new();

        for test_case in UTF8_TEST_CASES {
            let mut reader = parser::reader::Reader::new(test_case.binary.to_vec());
            match parser::parse(&module_registry, test_case.name, &mut reader) {
                Ok(_) => {
                    eprintln!(
                        "Test {} should have failed with '{}', but succeeded",
                        test_case.name, test_case.expected_error
                    );
                    failed += 1;
                }
                Err(e) => {
                    let error_string = e.to_string();
                    if error_string.contains(test_case.expected_error) {
                        passed += 1;
                    } else {
                        eprintln!(
                            "Test {} failed with '{}', expected '{}'",
                            test_case.name, error_string, test_case.expected_error
                        );
                        failed += 1;
                    }
                }
            }
        }

        println!("UTF-8 validation tests: {} passed, {} failed", passed, failed);
        assert_eq!(failed, 0, "{} UTF-8 tests failed", failed);
    }
}
