/* Execute corpus SQL on an in-memory database; emit tab-separated result rows. */
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int row(void *unused, int count, char **values, char **names) {
    (void)unused;
    (void)names;
    for (int i = 0; i < count; ++i) {
        if (i && putchar('\t') == EOF) return 1;
        if (fputs(values[i] ? values[i] : "NULL", stdout) == EOF) return 1;
    }
    return putchar('\n') == EOF;
}

int main(void) {
    sqlite3 *db = NULL;
    sqlite3_str *text = sqlite3_str_new(NULL);
    char buffer[4096];
    size_t size;
    while ((size = fread(buffer, 1, sizeof(buffer), stdin)) != 0) {
        if (memchr(buffer, '\0', size)) {
            fputs("SQL input contains a NUL byte\n", stderr);
            sqlite3_free(sqlite3_str_finish(text));
            return 1;
        }
        sqlite3_str_append(text, buffer, (int)size);
    }
    int error = sqlite3_str_errcode(text);
    char *sql = sqlite3_str_finish(text);
    if (ferror(stdin) || error != SQLITE_OK || !sql) {
        fputs("Unable to read SQL input\n", stderr);
        sqlite3_free(sql);
        return 1;
    }
    error = sqlite3_open(":memory:", &db);
    char *message = NULL;
    if (error == SQLITE_OK) error = sqlite3_exec(db, sql, row, NULL, &message);
    sqlite3_free(sql);
    if (error != SQLITE_OK) {
        fprintf(stderr, "SQLite: %s\n", message ? message : sqlite3_errmsg(db));
    }
    sqlite3_free(message);
    int close_error = sqlite3_close(db);
    return error != SQLITE_OK || close_error != SQLITE_OK || fflush(stdout) == EOF;
}
