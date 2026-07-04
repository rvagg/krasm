//! Lexer for WebAssembly Text Format.
//!
//! Tokenises WAT source into a stream of tokens. The lexer is implemented as
//! an iterator, producing tokens lazily on demand.
//!
//! # Example
//!
//! ```
//! use krasm::wat::Lexer;
//!
//! let source = "(module (func $add (param i32 i32) (result i32)))";
//! for result in Lexer::new(source) {
//!     let token = result.expect("valid token");
//!     println!("{:?}", token);
//! }
//! ```

use super::cursor::{Cursor, Position};
use super::error::LexError;
use super::token::{FloatLit, SignedValue, Token, TokenKind};

// ============================================================================
// Lexer
// ============================================================================

/// Lexer for WebAssembly Text Format.
///
/// Produces tokens via the `Iterator` trait. Each call to `next()` returns
/// the next token, or an error if the input is malformed.
pub struct Lexer<'a> {
    cursor: Cursor<'a>,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer for the given source text.
    pub fn new(source: &'a str) -> Self {
        Self {
            cursor: Cursor::new(source),
        }
    }

    /// Tokenise the entire source, returning all tokens or the first error.
    pub fn tokenise(source: &str) -> Result<Vec<Token>, LexError> {
        Lexer::new(source).collect()
    }

    /// Create an error at the given position.
    fn error(&self, message: impl Into<String>, pos: Position) -> LexError {
        LexError::new(message, pos.span_here())
    }

    /// Create an error spanning from start to current position.
    fn error_span(&self, message: impl Into<String>, start: Position) -> LexError {
        LexError::new(message, start.span_to(&self.cursor.position()))
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<Token, LexError>;

    fn next(&mut self) -> Option<Self::Item> {
        // Skip whitespace and comments
        if let Err(e) = self.skip_whitespace_and_comments() {
            return Some(Err(e));
        }
        if self.cursor.is_eof() {
            return None;
        }

        let start = self.cursor.position();
        let kind = match self.lex_token() {
            Ok(k) => k,
            Err(e) => return Some(Err(e)),
        };
        let span = start.span_to(&self.cursor.position());

        Some(Ok(Token::new(kind, span)))
    }
}

// ============================================================================
// Top-level token dispatch
// ============================================================================

impl<'a> Lexer<'a> {
    /// Lex a single token (after whitespace/comments have been skipped).
    fn lex_token(&mut self) -> Result<TokenKind, LexError> {
        let start = self.cursor.position();

        match self.cursor.peek().unwrap() {
            '(' => {
                self.cursor.advance();
                Ok(TokenKind::LeftParen)
            }
            ')' => {
                self.cursor.advance();
                Ok(TokenKind::RightParen)
            }
            '"' => {
                let result = self.lex_string()?;
                self.check_token_boundary(start)?;
                Ok(result)
            }
            '$' => {
                let result = self.lex_id()?;
                self.check_token_boundary(start)?;
                Ok(result)
            }
            '+' | '-' => self.lex_signed_number_or_keyword(),
            c if c.is_ascii_digit() => self.lex_number(false, false),
            c if is_idchar(c) => {
                let result = self.lex_keyword_or_special_float()?;
                self.check_token_boundary(start)?;
                Ok(result)
            }
            c => {
                self.cursor.advance();
                Err(self.error(format!("unexpected character: {:?}", c), start))
            }
        }
    }

    /// Lex a keyword, but check if it's actually a special float (inf, nan).
    fn lex_keyword_or_special_float(&mut self) -> Result<TokenKind, LexError> {
        let text = self.cursor.take_while(is_idchar);

        // Check for special float keywords
        if let Some(float) = parse_special_float(text, false) {
            return Ok(TokenKind::Float(float));
        }

        Ok(TokenKind::Keyword(text.to_string()))
    }

    /// Handle `+` or `-` prefix: could be a signed number or just a keyword.
    ///
    /// The heuristic checks the second character to decide:
    /// - Digit: definitely a number (`-42`)
    /// - `.`: likely a float (`-.5` → goes to `lex_decimal_float` via `lex_number`)
    /// - `i`/`n`: likely `inf`/`nan` (`-inf`, `-nan`)
    /// - Otherwise: treat as keyword (e.g., `+-` as an operator name)
    fn lex_signed_number_or_keyword(&mut self) -> Result<TokenKind, LexError> {
        let second = self.cursor.peek_second();

        // Check if this looks like a number based on the character after the sign
        let is_number = matches!(second, Some(c) if c.is_ascii_digit() || c == '.' || c == 'i' || c == 'n');

        if is_number {
            let negative = self.cursor.advance() == Some('-');
            self.lex_number(negative, true)
        } else {
            self.lex_keyword_or_special_float()
        }
    }
}

// ============================================================================
// Whitespace and comments
// ============================================================================

impl<'a> Lexer<'a> {
    /// Skip whitespace and comments. Returns error if comment is unterminated.
    fn skip_whitespace_and_comments(&mut self) -> Result<(), LexError> {
        loop {
            self.cursor.skip_while(|c| c.is_ascii_whitespace());

            if self.cursor.is_eof() {
                return Ok(());
            }

            match (self.cursor.peek(), self.cursor.peek_second()) {
                // Line comment: ;; to end of line
                (Some(';'), Some(';')) => {
                    self.cursor.skip_while(|c| c != '\n' && c != '\r');
                }
                // Block comment: (; ... ;) with nesting
                (Some('('), Some(';')) => {
                    self.skip_block_comment()?;
                }
                // Not a comment
                _ => return Ok(()),
            }
        }
    }

    /// Skip a block comment, handling nesting.
    fn skip_block_comment(&mut self) -> Result<(), LexError> {
        let start = self.cursor.position();

        // Consume opening "(;"
        self.cursor.advance();
        self.cursor.advance();

        let mut depth = 1;

        while depth > 0 {
            match (self.cursor.peek(), self.cursor.peek_second()) {
                (None, _) => {
                    return Err(self.error_span("unterminated block comment", start));
                }
                (Some('('), Some(';')) => {
                    self.cursor.advance();
                    self.cursor.advance();
                    depth += 1;
                }
                (Some(';'), Some(')')) => {
                    self.cursor.advance();
                    self.cursor.advance();
                    depth -= 1;
                }
                _ => {
                    self.cursor.advance();
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Identifiers and keywords
// ============================================================================

impl<'a> Lexer<'a> {
    /// Lex an identifier (starting with $).
    fn lex_id(&mut self) -> Result<TokenKind, LexError> {
        let start = self.cursor.position();

        // Consume the '$'
        self.cursor.advance();

        let name = self.cursor.take_while(is_idchar);
        if name.is_empty() {
            return Err(self.error_span("expected identifier after '$'", start));
        }

        Ok(TokenKind::Id(name.to_string()))
    }
}

// ============================================================================
// String literals
// ============================================================================

impl<'a> Lexer<'a> {
    /// Lex a string literal.
    fn lex_string(&mut self) -> Result<TokenKind, LexError> {
        let start = self.cursor.position();

        // Consume opening quote
        self.cursor.advance();

        let mut bytes = Vec::new();

        loop {
            match self.cursor.peek() {
                None => {
                    return Err(self.error_span("unterminated string literal", start));
                }
                Some('"') => {
                    self.cursor.advance();
                    break;
                }
                Some('\\') => {
                    self.cursor.advance();
                    self.lex_escape(&mut bytes)?;
                }
                Some(c) => {
                    self.cursor.advance();
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
        }

        Ok(TokenKind::String(bytes))
    }

    /// Lex an escape sequence (after the backslash), appending to `bytes`.
    fn lex_escape(&mut self, bytes: &mut Vec<u8>) -> Result<(), LexError> {
        let pos = self.cursor.position();
        let c = self
            .cursor
            .advance()
            .ok_or_else(|| self.error("unterminated escape sequence", pos))?;

        match c {
            't' => bytes.push(0x09),
            'n' => bytes.push(0x0A),
            'r' => bytes.push(0x0D),
            '"' => bytes.push(0x22),
            '\'' => bytes.push(0x27),
            '\\' => bytes.push(0x5C),
            'u' => self.lex_unicode_escape(bytes)?,
            c if c.is_ascii_hexdigit() => {
                let high = c.to_digit(16).unwrap() as u8;
                let low_pos = self.cursor.position();
                let low_char = self
                    .cursor
                    .advance()
                    .ok_or_else(|| self.error("unterminated hex escape", low_pos))?;
                let low = low_char
                    .to_digit(16)
                    .ok_or_else(|| self.error(format!("invalid hex digit: {:?}", low_char), low_pos))?
                    as u8;
                bytes.push((high << 4) | low);
            }
            _ => return Err(self.error(format!("invalid escape sequence: \\{}", c), pos)),
        }

        Ok(())
    }

    /// Lex a Unicode escape sequence \u{...}, appending UTF-8 bytes.
    fn lex_unicode_escape(&mut self, bytes: &mut Vec<u8>) -> Result<(), LexError> {
        let pos = self.cursor.position();

        // Expect opening brace
        match self.cursor.advance() {
            Some('{') => {}
            Some(c) => return Err(self.error(format!("expected '{{' after \\u, got {:?}", c), pos)),
            None => return Err(self.error("unterminated unicode escape", pos)),
        }

        // Collect hex digits
        let digits_start = self.cursor.position();
        let digits = self.cursor.take_while(|c| c.is_ascii_hexdigit());

        if digits.is_empty() {
            return Err(self.error("empty unicode escape", digits_start));
        }

        let code_point =
            u32::from_str_radix(digits, 16).map_err(|_| self.error("unicode escape value too large", digits_start))?;

        // Expect closing brace
        let close_pos = self.cursor.position();
        match self.cursor.advance() {
            Some('}') => {}
            Some(c) => return Err(self.error(format!("expected '}}' in unicode escape, got {:?}", c), close_pos)),
            None => return Err(self.error("unterminated unicode escape", close_pos)),
        }

        // Convert to UTF-8 bytes
        let c = char::from_u32(code_point)
            .ok_or_else(|| self.error(format!("invalid unicode code point: U+{:X}", code_point), digits_start))?;

        let mut buf = [0u8; 4];
        bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());

        Ok(())
    }
}

// ============================================================================
// Number literals
// ============================================================================

impl<'a> Lexer<'a> {
    /// Lex a number (integer or float). Sign has already been consumed if present.
    fn lex_number(&mut self, negative: bool, has_sign: bool) -> Result<TokenKind, LexError> {
        let start = self.cursor.position();

        // Check for special float values: inf, nan
        if matches!(self.cursor.peek(), Some('i') | Some('n')) {
            let result = self.lex_special_float(negative)?;
            self.check_token_boundary(start)?;
            return Ok(result);
        }

        // Check for hex prefix
        let is_hex = self.cursor.peek() == Some('0') && matches!(self.cursor.peek_second(), Some('x') | Some('X'));

        let result = if is_hex {
            self.cursor.advance(); // '0'
            self.cursor.advance(); // 'x'
            self.lex_hex_number(negative, has_sign)?
        } else {
            self.lex_decimal_number(negative, has_sign)?
        };

        self.check_token_boundary(start)?;
        Ok(result)
    }

    /// Verify the next character is a valid token boundary (whitespace, parens,
    /// comment start, or EOF). WAT requires whitespace or parentheses between
    /// all non-paren tokens. e.g. `1x`, `$l"a"`, and `"a""b"` are all invalid.
    fn check_token_boundary(&self, start: Position) -> Result<(), LexError> {
        match self.cursor.peek() {
            None => Ok(()),
            Some(c) if c.is_ascii_whitespace() => Ok(()),
            Some('(' | ')' | ';') => Ok(()),
            _ => Err(self.error("unknown operator", start)),
        }
    }

    /// Lex a hexadecimal number (after 0x prefix).
    fn lex_hex_number(&mut self, negative: bool, has_sign: bool) -> Result<TokenKind, LexError> {
        let start = self.cursor.position();
        let digits = self.cursor.take_while(|c| c.is_ascii_hexdigit() || c == '_');

        // Check for hex float (has '.' or 'p' exponent)
        if matches!(self.cursor.peek(), Some('.') | Some('p') | Some('P')) {
            return self.lex_float(negative, start, true);
        }

        if digits.is_empty() || digits == "_" {
            return Err(self.error("expected hex digits after '0x'", start));
        }

        if !validate_num_underscores(digits, true) {
            return Err(self.error("unknown operator", start));
        }

        let digits_clean: String = digits.chars().filter(|&c| c != '_').collect();
        match u64::from_str_radix(&digits_clean, 16) {
            Ok(value) => Ok(TokenKind::Integer(if has_sign {
                SignedValue::signed(value, negative)
            } else {
                SignedValue::unsigned(value)
            })),
            Err(_) => {
                // Overflow: emit as hex float so fhex handles precision correctly.
                // Large hex integers are only valid in float contexts (f32.const / f64.const).
                Ok(TokenKind::Float(FloatLit::Hex {
                    negative,
                    hex_str: format!("0x{}", digits_clean),
                }))
            }
        }
    }

    /// Lex a decimal number.
    fn lex_decimal_number(&mut self, negative: bool, has_sign: bool) -> Result<TokenKind, LexError> {
        let start = self.cursor.position();
        let digits = self.cursor.take_while(|c| c.is_ascii_digit() || c == '_');

        // Check for float (has '.' or 'e' exponent)
        if matches!(self.cursor.peek(), Some('.') | Some('e') | Some('E')) {
            return self.lex_float(negative, start, false);
        }

        if digits.is_empty() || digits == "_" {
            return Err(self.error("expected decimal digits", start));
        }

        if !validate_num_underscores(digits, false) {
            return Err(self.error("unknown operator", start));
        }

        let digits_clean: String = digits.chars().filter(|&c| c != '_').collect();
        match digits_clean.parse::<u64>() {
            Ok(value) => Ok(TokenKind::Integer(if has_sign {
                SignedValue::signed(value, negative)
            } else {
                SignedValue::unsigned(value)
            })),
            Err(_) => {
                // Overflow: emit as decimal float so str::parse handles precision correctly.
                // Large decimal integers are only valid in float contexts.
                Ok(TokenKind::Float(FloatLit::Decimal {
                    negative,
                    decimal_str: digits_clean,
                }))
            }
        }
    }

    /// Lex a float literal (decimal or hex). `start` is position of first digit.
    ///
    /// For hex floats, stores the original hex string as `FloatLit::Hex` so that
    /// f32 and f64 conversions can each round independently via `fhex::FromHex`.
    fn lex_float(&mut self, negative: bool, start: Position, hex: bool) -> Result<TokenKind, LexError> {
        // Consume fractional part if present
        if self.cursor.peek() == Some('.') {
            self.cursor.advance();
            if hex {
                self.cursor.skip_while(|c| c.is_ascii_hexdigit() || c == '_');
            } else {
                self.cursor.skip_while(|c| c.is_ascii_digit() || c == '_');
            }
        }

        // Consume exponent if present (p/P for hex, e/E for decimal)
        let has_exp = if hex {
            matches!(self.cursor.peek(), Some('p' | 'P'))
        } else {
            matches!(self.cursor.peek(), Some('e' | 'E'))
        };
        if has_exp {
            self.cursor.advance();
            if matches!(self.cursor.peek(), Some('+' | '-')) {
                self.cursor.advance();
            }
            let exp_digits = self.cursor.take_while(|c| c.is_ascii_digit() || c == '_');
            if !exp_digits.contains(|c: char| c.is_ascii_digit()) {
                return Err(self.error("unknown operator", start));
            }
        }

        let text = self.cursor.slice_from(&start);

        if !validate_num_underscores(text, hex) {
            return Err(self.error("unknown operator", start));
        }

        let text_clean: String = text.chars().filter(|&c| c != '_').collect();

        if hex {
            if !text_clean.chars().any(|c| c.is_ascii_hexdigit()) {
                return Err(self.error("invalid hex float", start));
            }
            Ok(TokenKind::Float(FloatLit::Hex {
                negative,
                hex_str: format!("0x{text_clean}"),
            }))
        } else {
            text_clean
                .parse::<f64>()
                .map_err(|_| self.error("invalid float literal", start))?;
            Ok(TokenKind::Float(FloatLit::Decimal {
                negative,
                decimal_str: text_clean,
            }))
        }
    }

    /// Lex special float values (inf, nan) after sign has been consumed.
    fn lex_special_float(&mut self, negative: bool) -> Result<TokenKind, LexError> {
        let start = self.cursor.position();
        let text = self
            .cursor
            .take_while(|c| c.is_ascii_alphanumeric() || c == ':' || c == '_');

        parse_special_float(text, negative)
            .map(TokenKind::Float)
            .ok_or_else(|| self.error(format!("invalid number: {}", text), start))
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Validate underscore placement in a numeric literal.
///
/// Per the WebAssembly spec, underscores in numeric literals may only appear
/// between two digits. Leading, trailing, or consecutive underscores are
/// rejected, as are underscores adjacent to non-digit characters like `.`,
/// `p`, `e`, `+`, or `-`.
///
/// For hex numbers, hex digits are valid around underscores in the
/// integer/fractional parts; the exponent part (after p/P) uses decimal digits.
fn validate_num_underscores(s: &str, is_hex: bool) -> bool {
    // Split at structural characters and validate each digit group
    let bytes = s.as_bytes();
    let hex_digit = |b: u8| -> bool { b.is_ascii_hexdigit() };
    let dec_digit = |b: u8| -> bool { b.is_ascii_digit() };

    // Walk through the string tracking which digit class is expected
    let mut in_exponent = false;

    for (i, &b) in bytes.iter().enumerate() {
        if b == b'_' {
            if i == 0 || i == bytes.len() - 1 {
                return false;
            }
            let check_digit: fn(u8) -> bool = if is_hex && !in_exponent { hex_digit } else { dec_digit };
            if !check_digit(bytes[i - 1]) || !check_digit(bytes[i + 1]) {
                return false;
            }
        } else if b == b'p' || b == b'P' || (!is_hex && (b == b'e' || b == b'E')) {
            in_exponent = true;
        }
    }
    true
}

/// Check if a character is valid in a WAT identifier.
///
/// Per the WebAssembly text format spec, identifiers (idchar) can contain:
/// - ASCII letters and digits
/// - Many punctuation characters: `!#$%&'*+-./:<=>?@\^_`|~`
///
/// This permissive set allows identifiers like `$my_func`, `$add/sub`, `$a.b.c`.
/// See: <https://webassembly.github.io/spec/core/text/values.html#text-id>
fn is_idchar(c: char) -> bool {
    matches!(
        c,
        '0'..='9'
            | 'a'..='z'
            | 'A'..='Z'
            | '!'
            | '#'
            | '$'
            | '%'
            | '&'
            | '\''
            | '*'
            | '+'
            | '-'
            | '.'
            | '/'
            | ':'
            | '<'
            | '='
            | '>'
            | '?'
            | '@'
            | '\\'
            | '^'
            | '_'
            | '`'
            | '|'
            | '~'
    )
}

/// Parse special float keywords: inf, nan, nan:0x...
fn parse_special_float(text: &str, negative: bool) -> Option<FloatLit> {
    if text == "inf" {
        Some(FloatLit::Inf { negative })
    } else if text == "nan" {
        Some(FloatLit::Nan {
            negative,
            payload: None,
        })
    } else if let Some(payload_hex) = text.strip_prefix("nan:0x") {
        if payload_hex.is_empty() || !validate_num_underscores(payload_hex, true) {
            return None;
        }
        let clean: String = payload_hex.chars().filter(|&c| c != '_').collect();
        let payload = u64::from_str_radix(&clean, 16).ok()?;
        Some(FloatLit::Nan {
            negative,
            payload: Some(payload),
        })
    } else {
        None
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::token::Span;
    use super::*;

    /// Helper to tokenise and extract just the token kinds.
    fn kinds(source: &str) -> Vec<TokenKind> {
        Lexer::tokenise(source)
            .expect("tokenise failed")
            .into_iter()
            .map(|t| t.kind)
            .collect()
    }

    /// Helper to check that lexing fails with an error containing the substring.
    fn expect_error(source: &str, substring: &str) {
        let err = Lexer::tokenise(source).expect_err("expected error");
        assert!(
            err.message.contains(substring),
            "Expected error containing {:?}, got {:?}",
            substring,
            err.message
        );
    }

    /// Helper to create an unsigned integer token kind (bare literal).
    fn int(value: u64) -> TokenKind {
        TokenKind::Integer(SignedValue::unsigned(value))
    }

    /// Helper to create an explicitly signed integer token kind (+N or -N).
    fn signed_int(value: u64, negative: bool) -> TokenKind {
        TokenKind::Integer(SignedValue::signed(value, negative))
    }

    // =========================================================================
    // Parentheses
    // =========================================================================

    #[test]
    fn empty_input() {
        assert_eq!(kinds(""), vec![]);
    }

    #[test]
    fn just_parens() {
        assert_eq!(kinds("()"), vec![TokenKind::LeftParen, TokenKind::RightParen]);
    }

    #[test]
    fn nested_parens() {
        assert_eq!(
            kinds("(())"),
            vec![
                TokenKind::LeftParen,
                TokenKind::LeftParen,
                TokenKind::RightParen,
                TokenKind::RightParen,
            ]
        );
    }

    #[test]
    fn parens_with_whitespace() {
        assert_eq!(kinds("  (  )  "), vec![TokenKind::LeftParen, TokenKind::RightParen]);
    }

    // =========================================================================
    // Keywords
    // =========================================================================

    #[test]
    fn simple_keywords() {
        assert_eq!(
            kinds("module func param"),
            vec![
                TokenKind::Keyword("module".into()),
                TokenKind::Keyword("func".into()),
                TokenKind::Keyword("param".into()),
            ]
        );
    }

    #[test]
    fn dotted_keywords() {
        assert_eq!(
            kinds("i32.const i32.add memory.grow"),
            vec![
                TokenKind::Keyword("i32.const".into()),
                TokenKind::Keyword("i32.add".into()),
                TokenKind::Keyword("memory.grow".into()),
            ]
        );
    }

    #[test]
    fn keyword_with_numbers() {
        assert_eq!(
            kinds("i32 i64 f32 f64 v128"),
            vec![
                TokenKind::Keyword("i32".into()),
                TokenKind::Keyword("i64".into()),
                TokenKind::Keyword("f32".into()),
                TokenKind::Keyword("f64".into()),
                TokenKind::Keyword("v128".into()),
            ]
        );
    }

    // =========================================================================
    // Identifiers
    // =========================================================================

    #[test]
    fn simple_ids() {
        assert_eq!(
            kinds("$foo $bar"),
            vec![TokenKind::Id("foo".into()), TokenKind::Id("bar".into())]
        );
    }

    #[test]
    fn numeric_id() {
        assert_eq!(
            kinds("$0 $123"),
            vec![TokenKind::Id("0".into()), TokenKind::Id("123".into())]
        );
    }

    #[test]
    fn id_with_special_chars() {
        assert_eq!(
            kinds("$my_func $add/sub $a.b.c"),
            vec![
                TokenKind::Id("my_func".into()),
                TokenKind::Id("add/sub".into()),
                TokenKind::Id("a.b.c".into()),
            ]
        );
    }

    #[test]
    fn empty_id_error() {
        expect_error("$", "expected identifier");
    }

    // =========================================================================
    // Strings
    // =========================================================================

    #[test]
    fn empty_string() {
        assert_eq!(kinds(r#""""#), vec![TokenKind::String(vec![])]);
    }

    #[test]
    fn simple_string() {
        assert_eq!(kinds(r#""hello""#), vec![TokenKind::String(b"hello".to_vec())]);
    }

    #[test]
    fn string_with_simple_escapes() {
        assert_eq!(kinds(r#""\t\n\r""#), vec![TokenKind::String(vec![0x09, 0x0A, 0x0D])]);
    }

    #[test]
    fn string_with_quote_escapes() {
        assert_eq!(kinds(r#""\"\'\\""#), vec![TokenKind::String(vec![0x22, 0x27, 0x5C])]);
    }

    #[test]
    fn string_with_hex_escapes() {
        assert_eq!(kinds(r#""\00\ff\42""#), vec![TokenKind::String(vec![0x00, 0xFF, 0x42])]);
    }

    #[test]
    fn string_with_unicode_escape() {
        assert_eq!(
            kinds(r#""\u{1F600}""#),
            vec![TokenKind::String("😀".as_bytes().to_vec())]
        );
    }

    #[test]
    fn unterminated_string() {
        expect_error(r#""hello"#, "unterminated");
    }

    #[test]
    fn invalid_escape() {
        expect_error(r#""\z""#, "invalid escape");
    }

    // =========================================================================
    // Integers
    // =========================================================================

    #[test]
    fn decimal_integers() {
        assert_eq!(kinds("0 42 123"), vec![int(0), int(42), int(123)]);
    }

    #[test]
    fn signed_integers() {
        assert_eq!(
            kinds("-1 +42 -0"),
            vec![signed_int(1, true), signed_int(42, false), signed_int(0, true)]
        );
    }

    #[test]
    fn hex_integers() {
        assert_eq!(kinds("0x0 0xDEAD 0xff"), vec![int(0), int(0xDEAD), int(0xFF)]);
    }

    #[test]
    fn signed_hex() {
        assert_eq!(kinds("-0x10"), vec![signed_int(0x10, true)]);
    }

    #[test]
    fn integer_with_underscores() {
        assert_eq!(kinds("1_000_000"), vec![int(1_000_000)]);
        assert_eq!(kinds("0xFF_FF"), vec![int(0xFFFF)]);
    }

    #[test]
    fn max_u64() {
        // This would overflow with i64, but works with our representation
        let tokens = kinds("0xFFFFFFFFFFFFFFFF");
        assert_eq!(tokens, vec![int(u64::MAX)]);

        // Verify it converts correctly
        if let TokenKind::Integer(sv) = &tokens[0] {
            assert_eq!(sv.to_u64(), Some(u64::MAX));
            assert_eq!(sv.to_i64(), None); // Too large for i64
        }
    }

    #[test]
    fn mixed_case_hex() {
        assert_eq!(kinds("0xAbCd"), vec![int(0xABCD)]);
        assert_eq!(kinds("0xDeAdBeEf"), vec![int(0xDEADBEEF)]);
    }

    #[test]
    fn consecutive_underscores() {
        // Spec forbids consecutive underscores: underscore must be between two digits
        expect_error("1__2", "unknown operator");
        expect_error("0x1__f", "unknown operator");
    }

    #[test]
    fn invalid_hex_forms() {
        // 0x. is invalid - needs at least one hex digit
        expect_error("0x.", "invalid hex float");
        // 0x alone is invalid
        expect_error("0x", "expected hex digits");
        // 0x_ is invalid (only underscore, no digits)
        expect_error("0x_", "expected hex digits");
    }

    // =========================================================================
    // Floats
    // =========================================================================

    #[test]
    fn simple_floats() {
        assert_eq!(
            kinds("3.14"),
            vec![TokenKind::Float(FloatLit::Decimal {
                negative: false,
                decimal_str: "3.14".into()
            })]
        );
        assert_eq!(
            kinds("0.5"),
            vec![TokenKind::Float(FloatLit::Decimal {
                negative: false,
                decimal_str: "0.5".into()
            })]
        );
    }

    #[test]
    fn float_with_exponent() {
        assert_eq!(
            kinds("1e10"),
            vec![TokenKind::Float(FloatLit::Decimal {
                negative: false,
                decimal_str: "1e10".into()
            })]
        );
        assert_eq!(
            kinds("1.5e-3"),
            vec![TokenKind::Float(FloatLit::Decimal {
                negative: false,
                decimal_str: "1.5e-3".into()
            })]
        );
    }

    #[test]
    fn special_floats() {
        assert_eq!(kinds("inf"), vec![TokenKind::Float(FloatLit::Inf { negative: false })]);
        assert_eq!(kinds("-inf"), vec![TokenKind::Float(FloatLit::Inf { negative: true })]);
        assert_eq!(
            kinds("nan"),
            vec![TokenKind::Float(FloatLit::Nan {
                negative: false,
                payload: None
            })]
        );
        assert_eq!(
            kinds("-nan"),
            vec![TokenKind::Float(FloatLit::Nan {
                negative: true,
                payload: None
            })]
        );
    }

    #[test]
    fn nan_with_payload() {
        assert_eq!(
            kinds("nan:0x1234"),
            vec![TokenKind::Float(FloatLit::Nan {
                negative: false,
                payload: Some(0x1234)
            })]
        );
        assert_eq!(
            kinds("-nan:0xABCD"),
            vec![TokenKind::Float(FloatLit::Nan {
                negative: true,
                payload: Some(0xABCD)
            })]
        );
        // Underscores in NaN payload
        assert_eq!(
            kinds("nan:0x7f_ffff"),
            vec![TokenKind::Float(FloatLit::Nan {
                negative: false,
                payload: Some(0x7f_ffff)
            })]
        );
        assert_eq!(
            kinds("nan:0xf_ffff_ffff_ffff"),
            vec![TokenKind::Float(FloatLit::Nan {
                negative: false,
                payload: Some(0xf_ffff_ffff_ffff)
            })]
        );
    }

    #[test]
    fn hex_float() {
        // 0x1.8p1 = 1.5 * 2 = 3.0
        if let [TokenKind::Float(fl)] = kinds("0x1.8p1").as_slice() {
            assert!((fl.to_f64() - 3.0).abs() < 0.0001);
            assert!((fl.to_f32() - 3.0f32).abs() < 0.0001);
        } else {
            panic!("expected hex float");
        }
    }

    #[test]
    fn hex_integer_with_underscores() {
        // Valid underscore placement: between hex digits
        assert_eq!(
            kinds("0x0125_6789_ADEF_bcef"),
            vec![TokenKind::Integer(SignedValue::unsigned(0x01256789ADEFBCEF))]
        );
        // Valid: simple hex with underscore
        assert_eq!(kinds("0x1_0"), vec![TokenKind::Integer(SignedValue::unsigned(0x10))]);
    }

    #[test]
    fn underscore_validation() {
        // Invalid: trailing underscore
        assert!(Lexer::tokenise("0x100_").is_err());
        // Invalid: leading underscore after 0x
        assert!(Lexer::tokenise("0x_100").is_err());
        // Invalid: underscore adjacent to dot
        assert!(Lexer::tokenise("0x1_.0p1").is_err());
        assert!(Lexer::tokenise("0x1._0p1").is_err());
        // Invalid: underscore adjacent to p
        assert!(Lexer::tokenise("0x1.0_p1").is_err());
        assert!(Lexer::tokenise("0x1.0p_1").is_err());
        // Invalid: decimal trailing underscore
        assert!(Lexer::tokenise("100_").is_err());
        // Invalid: decimal leading underscore
        // Note: _100 is lexed as a keyword, not a number
    }

    // =========================================================================
    // Comments
    // =========================================================================

    #[test]
    fn line_comment() {
        assert_eq!(
            kinds("foo ;; this is a comment\nbar"),
            vec![TokenKind::Keyword("foo".into()), TokenKind::Keyword("bar".into())]
        );
    }

    #[test]
    fn line_comment_at_eof() {
        assert_eq!(kinds("foo ;; comment"), vec![TokenKind::Keyword("foo".into())]);
    }

    #[test]
    fn block_comment() {
        assert_eq!(
            kinds("foo (; comment ;) bar"),
            vec![TokenKind::Keyword("foo".into()), TokenKind::Keyword("bar".into())]
        );
    }

    #[test]
    fn nested_block_comment() {
        assert_eq!(
            kinds("foo (; outer (; inner ;) outer ;) bar"),
            vec![TokenKind::Keyword("foo".into()), TokenKind::Keyword("bar".into())]
        );
    }

    #[test]
    fn block_comment_with_newlines() {
        assert_eq!(
            kinds("foo (;\nmulti\nline\n;) bar"),
            vec![TokenKind::Keyword("foo".into()), TokenKind::Keyword("bar".into())]
        );
    }

    #[test]
    fn unterminated_block_comment() {
        expect_error("(; unterminated", "unterminated block comment");
    }

    #[test]
    fn unicode_in_comments() {
        // Unicode should be allowed in comments
        assert_eq!(
            kinds("foo ;; こんにちは 🎉\nbar"),
            vec![TokenKind::Keyword("foo".into()), TokenKind::Keyword("bar".into())]
        );
        assert_eq!(
            kinds("foo (; émoji: 🦀 ;) bar"),
            vec![TokenKind::Keyword("foo".into()), TokenKind::Keyword("bar".into())]
        );
    }

    // =========================================================================
    // Spans
    // =========================================================================

    #[test]
    fn span_accuracy() {
        let tokens = Lexer::tokenise("(module)").unwrap();

        assert_eq!(tokens[0].span, Span::new(0, 1, 1, 1)); // (
        assert_eq!(tokens[1].span, Span::new(1, 7, 1, 2)); // module
        assert_eq!(tokens[2].span, Span::new(7, 8, 1, 8)); // )
    }

    #[test]
    fn multiline_spans() {
        let tokens = Lexer::tokenise("(\n  module\n)").unwrap();

        assert_eq!(tokens[0].span.line, 1); // (
        assert_eq!(tokens[1].span.line, 2); // module
        assert_eq!(tokens[2].span.line, 3); // )
    }

    // =========================================================================
    // Integration tests
    // =========================================================================

    #[test]
    fn simple_module() {
        let wat = "(module)";
        let tokens = kinds(wat);
        assert_eq!(
            tokens,
            vec![
                TokenKind::LeftParen,
                TokenKind::Keyword("module".into()),
                TokenKind::RightParen,
            ]
        );
    }

    #[test]
    fn function_with_params() {
        let wat = "(func $add (param $a i32) (param $b i32) (result i32))";
        let tokens = kinds(wat);
        assert!(tokens.contains(&TokenKind::Keyword("func".into())));
        assert!(tokens.contains(&TokenKind::Id("add".into())));
        assert!(tokens.contains(&TokenKind::Keyword("param".into())));
        assert!(tokens.contains(&TokenKind::Id("a".into())));
        assert!(tokens.contains(&TokenKind::Keyword("i32".into())));
        assert!(tokens.contains(&TokenKind::Keyword("result".into())));
    }

    #[test]
    fn data_segment() {
        let wat = r#"(data (i32.const 8) "Hello\n")"#;
        let tokens = kinds(wat);
        assert!(tokens.contains(&TokenKind::Keyword("data".into())));
        assert!(tokens.contains(&int(8)));
        assert!(tokens.contains(&TokenKind::String(b"Hello\n".to_vec())));
    }

    #[test]
    fn real_wat_fib_iterative() {
        let wat = r#"
;; Iterative Fibonacci
(module
  (func (export "fib") (param $n i32) (result i32)
    (local $a i32)
    (local $b i32)
    (if (i32.eqz (local.get $n))
      (then (return (i32.const 0)))
    )
    (local.set $a (i32.const 0))
    (local.set $b (i32.const 1))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
        (local.set $b (i32.add (local.get $a) (local.get $b)))
        (br $loop)
      )
    )
    (local.get $b)
  )
)
"#;
        let tokens = Lexer::tokenise(wat).expect("should tokenise");
        assert!(tokens.len() > 50, "expected many tokens, got {}", tokens.len());

        // Spot check some tokens
        let has_keyword = |k: &str| tokens.iter().any(|t| t.kind == TokenKind::Keyword(k.into()));
        let has_id = |k: &str| tokens.iter().any(|t| t.kind == TokenKind::Id(k.into()));

        assert!(has_keyword("module"));
        assert!(has_keyword("func"));
        assert!(has_keyword("i32.eqz"));
        assert!(has_keyword("local.get"));
        assert!(has_keyword("br_if"));
        assert!(has_id("n"));
        assert!(has_id("done"));
        assert!(has_id("loop"));
    }
}

// ============================================================================
// Property-based tests
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Token spans must be valid: within bounds and non-inverted.
        #[test]
        fn spans_are_valid(source in "\\PC{0,200}") {
            for token in Lexer::new(&source).flatten() {
                prop_assert!(
                    token.span.start <= token.span.end,
                    "span inverted: start {} > end {}",
                    token.span.start, token.span.end
                );
                prop_assert!(
                    token.span.end <= source.len(),
                    "span end {} exceeds source length {}",
                    token.span.end, source.len()
                );
            }
        }

        /// Token text extraction must be valid UTF-8 and match the span.
        #[test]
        fn token_text_is_valid(source in "\\PC{0,200}") {
            for token in Lexer::new(&source).flatten() {
                let text = token.text(&source);
                prop_assert_eq!(
                    text.len(),
                    token.span.end - token.span.start,
                    "text length {} doesn't match span length {}",
                    text.len(), token.span.end - token.span.start
                );
            }
        }

        /// Successful tokens must not overlap.
        #[test]
        fn tokens_do_not_overlap(source in "\\PC{0,200}") {
            let tokens: Vec<_> = Lexer::new(&source)
                .filter_map(Result::ok)
                .collect();

            for window in tokens.windows(2) {
                prop_assert!(
                    window[0].span.end <= window[1].span.start,
                    "tokens overlap: first ends at {}, second starts at {}",
                    window[0].span.end, window[1].span.start
                );
            }
        }

        /// Keywords extracted via text() should match the keyword string.
        #[test]
        fn keyword_text_matches(source in "[a-z][a-z0-9.]{0,20}( [a-z][a-z0-9.]{0,20}){0,5}") {
            for token in Lexer::new(&source).flatten() {
                if let TokenKind::Keyword(ref kw) = token.kind {
                    let text = token.text(&source);
                    prop_assert_eq!(
                        text, kw.as_str(),
                        "keyword text '{}' doesn't match TokenKind '{}'",
                        text, kw
                    );
                }
            }
        }

        /// Identifiers extracted via text() should be '$' + the id string.
        #[test]
        fn id_text_matches(source in "\\$[a-z_][a-z0-9_]{0,10}( \\$[a-z_][a-z0-9_]{0,10}){0,5}") {
            for token in Lexer::new(&source).flatten() {
                if let TokenKind::Id(ref id) = token.kind {
                    let text = token.text(&source);
                    let expected = format!("${}", id);
                    prop_assert_eq!(
                        text, expected.as_str(),
                        "id text '{}' doesn't match expected '{}'",
                        text, expected
                    );
                }
            }
        }

        /// Line numbers must be monotonically non-decreasing.
        #[test]
        fn line_numbers_increase(source in "[a-z0-9()\\n ]{0,100}") {
            let mut last_line = 0u32;
            for token in Lexer::new(&source).flatten() {
                prop_assert!(
                    token.span.line >= last_line,
                    "line number decreased: {} -> {}",
                    last_line, token.span.line
                );
                last_line = token.span.line;
            }
        }

        /// The lexer must never panic on arbitrary input.
        /// (This complements fuzzing with deterministic, reproducible tests.)
        #[test]
        fn never_panics(source in "\\PC{0,500}") {
            // Just iterate through - if we get here without panic, we pass
            for result in Lexer::new(&source) {
                let _ = result;
            }
        }

        /// Integer tokens must round-trip through Display correctly.
        #[test]
        fn integer_display_roundtrip(
            value in 0u64..=u64::MAX,
            negative in proptest::bool::ANY
        ) {
            let sv = if negative {
                SignedValue::signed(value, true)
            } else {
                SignedValue::unsigned(value)
            };
            let displayed = format!("{}", TokenKind::Integer(sv));

            // Parse it back (simplified check)
            if negative && value > 0 {
                prop_assert!(displayed.starts_with('-'));
            } else {
                prop_assert!(!displayed.starts_with('-') || value == 0);
            }
        }
    }
}
