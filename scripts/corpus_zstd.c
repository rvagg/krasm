/* Stream one level-3 frame or decode frames through standard input/output. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

int main(int argc, char **argv) {
    if (argc != 2 || (strcmp(argv[1], "compress") && strcmp(argv[1], "decompress"))) {
        fputs("Usage: zstd compress|decompress\n", stderr);
        return 1;
    }
    int compress = !strcmp(argv[1], "compress");
    size_t input_size = compress ? ZSTD_CStreamInSize() : ZSTD_DStreamInSize();
    size_t output_size = compress ? ZSTD_CStreamOutSize() : ZSTD_DStreamOutSize();
    void *input = malloc(input_size);
    void *output = malloc(output_size);
    ZSTD_CCtx *encoder = compress ? ZSTD_createCCtx() : NULL;
    ZSTD_DCtx *decoder = compress ? NULL : ZSTD_createDCtx();
    int status = 1;
    if (!input || !output || (compress ? !encoder : !decoder)) {
        fputs("Unable to allocate streaming buffers\n", stderr);
        goto done;
    }
    if (compress) {
        size_t result = ZSTD_CCtx_setParameter(encoder, ZSTD_c_compressionLevel, 3);
        if (ZSTD_isError(result)) {
            fprintf(stderr, "zstd: %s\n", ZSTD_getErrorName(result));
            goto done;
        }
    }
    size_t remaining = 1;
    for (;;) {
        size_t count = fread(input, 1, input_size, stdin);
        if (ferror(stdin)) {
            fputs("Unable to read input\n", stderr);
            goto done;
        }
        int end = feof(stdin);
        ZSTD_inBuffer in = {input, count, 0};
        if (!compress && count == 0) {
            if (remaining != 0) {
                fputs("Incomplete zstd frame\n", stderr);
                goto done;
            }
            break;
        }
        do {
            ZSTD_outBuffer out = {output, output_size, 0};
            remaining = compress
                ? ZSTD_compressStream2(encoder, &out, &in, end ? ZSTD_e_end : ZSTD_e_continue)
                : ZSTD_decompressStream(decoder, &out, &in);
            if (ZSTD_isError(remaining)) {
                fprintf(stderr, "zstd: %s\n", ZSTD_getErrorName(remaining));
                goto done;
            }
            if (fwrite(output, 1, out.pos, stdout) != out.pos) {
                fputs("Unable to write output\n", stderr);
                goto done;
            }
        } while (in.pos < in.size || (compress && end && remaining != 0));
        if (compress && end) break;
    }
    status = fflush(stdout) == EOF;
done:
    ZSTD_freeCCtx(encoder);
    ZSTD_freeDCtx(decoder);
    free(input);
    free(output);
    return status;
}
