#!/bin/sh
#
# Download the sparse test matrices used by the tests, into the
# current working directory (normally build/test).
#
# Every matrix is first fetched from the SuiteSparse collection. Any
# matrix that could not be fetched there is taken from the STRUMPACK
# test matrix bundle at NERSC, which contains most (not all) of them.
#
# A matrix only counts as present when <name>/<name>.mtx exists and is
# non-empty; broken or partial downloads are removed.
#
# Exit codes:
#   0   all matrices are available
#   77  at least one matrix is missing. ctest treats this as "skipped"
#       (SKIP_RETURN_CODE), and the tests that need a missing matrix
#       are skipped too (see test/require_mtx.sh).
#
# This script never exits with a failure because of a network problem.

# group/name in the SuiteSparse collection
MATRICES="
TOKAMAK/utm300
Pothen/mesh3e1
Oberwolfach/t2dal
HB/bcsstk28
DRIVCAV/cavity16
HB/bcsstm08
HB/sherman4
Bai/rdb968
HB/gemat11
"

SUITESPARSE_URL="https://suitesparse-collection-website.herokuapp.com/MM"
# Bundle with utm300 mesh3e1 t2dal bcsstk28 cavity16 bcsstm08 sherman4 rdb968
NERSC_BUNDLE_URL="https://portal.nersc.gov/project/sparse/strumpack/test_matrices.tar.gz"

# download <url> <output file>, returns non-zero on failure
download() {
    rm -f "$2"
    if command -v wget >/dev/null 2>&1; then
        wget -q --tries=3 --timeout=30 --waitretry=5 -O "$2" "$1"
    elif command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 --retry-delay 5 --connect-timeout 30 \
             --max-time 300 -o "$2" "$1"
    else
        echo "  neither wget nor curl is available"
        return 1
    fi
    status=$?
    # wget leaves an empty file behind on HTTP errors
    if [ $status -ne 0 ] || [ ! -s "$2" ]; then
        rm -f "$2"
        return 1
    fi
    return 0
}

has_matrix() {
    [ -s "$1/$1.mtx" ]
}

# extract <tarball> [members...], only if it is a valid gzip'ed tar
extract() {
    tarball=$1
    shift
    if ! tar -tzf "$tarball" >/dev/null 2>&1; then
        echo "  $tarball is not a valid .tar.gz archive"
        return 1
    fi
    tar -xzf "$tarball" "$@" >/dev/null 2>&1
}

missing() {
    for m in $MATRICES; do
        name=${m#*/}
        has_matrix "$name" || echo "$name"
    done
}

# 1) individual matrices from the SuiteSparse collection
for m in $MATRICES; do
    name=${m#*/}
    if has_matrix "$name"; then
        echo "$name: already present"
        continue
    fi
    echo "$name: downloading from SuiteSparse"
    rm -rf "$name"
    if download "$SUITESPARSE_URL/$m.tar.gz" "$name.tar.gz"; then
        extract "$name.tar.gz"
    else
        echo "  download failed"
    fi
    rm -f "$name.tar.gz"
    if ! has_matrix "$name"; then
        rm -rf "$name"
    fi
done

# 2) fallback: NERSC bundle, for whatever is still missing
todo=$(missing)
if [ -n "$todo" ]; then
    echo "Still missing: $(echo $todo); trying the NERSC bundle"
    if download "$NERSC_BUNDLE_URL" test_matrices.tar.gz; then
        if tar -tzf test_matrices.tar.gz >/dev/null 2>&1; then
            for name in $todo; do
                if tar -tzf test_matrices.tar.gz | grep -q "^$name/$name.mtx\$"; then
                    rm -rf "$name"
                    extract test_matrices.tar.gz "$name/" || true
                    has_matrix "$name" && echo "$name: taken from NERSC bundle"
                    has_matrix "$name" || rm -rf "$name"
                fi
            done
        else
            echo "  NERSC bundle is not a valid .tar.gz archive"
        fi
    else
        echo "  download failed"
    fi
    rm -f test_matrices.tar.gz
fi

todo=$(missing)
if [ -n "$todo" ]; then
    echo "WARNING: could not download test matrices: $(echo $todo)"
    echo "         tests using these matrices will be skipped"
    exit 77
fi
echo "All test matrices are available"
exit 0
