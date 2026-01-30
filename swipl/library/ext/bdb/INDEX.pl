/*  Creator: make/0

    Purpose: Provide index for autoload
*/

index(bdb_init(?), bdb, bdb).
index(bdb_init(?,?), bdb, bdb).
index(bdb_close_environment(?), bdb, bdb).
index(bdb_current_environment(?), bdb, bdb).
index(bdb_environment_property(?,?), bdb, bdb).
index(bdb_open(?,?,?,?), bdb, bdb).
index(bdb_close(?), bdb, bdb).
index(bdb_closeall, bdb, bdb).
index(bdb_current(?), bdb, bdb).
index(bdb_put(?,?,?), bdb, bdb).
index(bdb_del(?,?,?), bdb, bdb).
index(bdb_delall(?,?,?), bdb, bdb).
index(bdb_enum(?,?,?), bdb, bdb).
index(bdb_get(?,?,?), bdb, bdb).
index(bdb_getall(?,?,?), bdb, bdb).
index(bdb_transaction(0), bdb, bdb).
index(bdb_transaction(+,0), bdb, bdb).
index(bdb_version(?), bdb, bdb).
