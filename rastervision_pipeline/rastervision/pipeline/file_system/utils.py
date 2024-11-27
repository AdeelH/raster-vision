from typing import TYPE_CHECKING, Any
import os
from os.path import abspath, basename, join
import shutil
from threading import Timer
import time
import logging
import json
import zipfile
from urllib.parse import urlparse

from tqdm.auto import tqdm
import fsspec
from fsspec import AbstractFileSystem as FileSystem

from rastervision.pipeline import rv_config_ as rv_config
# from rastervision.pipeline.file_system import FileSystem
# from rastervision.pipeline.file_system.local_file_system import (
#     LocalFileSystem, make_dir)

if TYPE_CHECKING:
    from tempfile import TemporaryDirectory

log = logging.getLogger(__name__)


def get_file_system(uri: str) -> FileSystem:
    fs, _ = fsspec.url_to_fs(uri)
    return fs


def sync_to_dir(src_dir: str, dst_dir_uri: str,
                fs: FileSystem | None = None) -> None:  # pragma: no cover
    """Synchronize a local source directory to destination directory.

    Transfers files from source to destination directories so that the
    destination has all the source files. If FileSystem is remote, this
    involves uploading.

    Args:
        src_dir: path of local source directory
        dst_dir_uri: URI of destination directory
        fs: if supplied, use fs instead of automatically chosen FileSystem for
            dst_dir_uri
    """
    if not fs:
        fs = get_file_system(dst_dir_uri)
    fs.put(src_dir, dst_dir_uri, recursive=True)


def sync_from_dir(src_dir_uri: str, dst_dir: str,
                  fs: FileSystem | None = None):
    """Synchronize a source directory to local destination directory.

    Transfers files from source to destination directories so that the
    destination has all the source files. If FileSystem is remote, this
    involves downloading.

    Args:
        src_dir_uri: URI of source directory
        dst_dir: path of local destination directory
        fs: if supplied, use fs instead of automatically chosen FileSystem for
            dst_dir_uri
    """
    if not fs:
        fs = get_file_system(src_dir_uri)
    fs.get(src_dir_uri, dst_dir)


def download_if_needed(uri: str,
                       download_dir: str | None = None,
                       fs: FileSystem | None = None,
                       use_cache: bool = True) -> str:
    """Download a file to a directory if remote and return its local path.

    The full local path, within ``download_dir``, is determined by
    :func:`.get_local_path`. If a file doesn't already exists at that path, it
    is downloaded.

    Args:
        uri: URI of file to download. If this is a local path, it will be
            returned as is.
        download_dir: Local directory to download file into. If ``None``, the
            file will be downloaded to cache dir as defined by ``RVConfig``.
            Defaults to ``None``.
        fs: If provided, use ``fs`` instead of the automatically chosen
            :class:`.FileSystem` for ``uri``. Defaults to ``None``.
        use_cache: If ``False`` and the file is remote, download it regardless
            of whether it exists in cache. Defaults to ``True``.

    Returns:
        Local path to file.

    Raises:
        NotReadableError: if URI cannot be read from.
    """
    if download_dir is None:
        download_dir = rv_config.get_cache_dir()

    if not fs:
        fs = get_file_system(uri)

    local_path = get_local_path(uri, download_dir, fs=fs)
    if local_path == uri:
        return local_path

    if use_cache and file_exists(local_path, include_dir=False):
        log.info(f'Using cached file {local_path}.')
        return local_path

    log.info(f'Downloading {uri} to {local_path}...')
    make_dir(local_path, use_dirname=True)
    fs.copy_from(uri, local_path)

    return local_path


def download_or_copy(uri: str, target_dir: str,
                     fs: FileSystem | None = None) -> str:
    """Download or copy a file to a directory and return the local file path.

    If the file already exists in ``target_dir``, nothing is done. If the file
    is elsewhere but still local, it is copied to ``target_dir``. If it is
    remote, it is downloaded to the cache dir and then moved to ``target_dir``.

    Args:
        uri: URI of file.
        target_dir: Local directory to download or copy file to.
        fs: If supplied, use fs instead of automatically chosen
            :class:`.FileSystem` for ``uri``.

    Returns:
        Local path to file.
    """
    target_path = join(target_dir, basename(uri))
    if file_exists(target_path, fs=LocalFileSystem, include_dir=False):
        pass
    elif is_local(uri):
        shutil.copy(uri, target_path)
    else:
        download_path = download_if_needed(uri, fs=fs)
        shutil.move(download_path, target_path)
    return target_path


def file_exists(uri: str,
                fs: FileSystem | None = None,
                include_dir: bool = True) -> bool:
    """Check if file exists.

    Args:
        uri: URI of file
        fs: If supplied, use fs instead of automatically chosen
            :class:`.FileSystem` for ``uri``.
        include_dir: Include directories in check, if the file system
            supports directory reads. Otherwise only return true if a single
            file exists at the URI. Defaults to ``True``.
    """
    if not fs:
        fs = get_file_system(uri)
    if include_dir:
        return fs.exists(uri)
    return fs.isfile(uri)


def list_paths(uri: str, ext: str = '', fs: FileSystem | None = None,
               **kwargs) -> list[str]:
    """List paths rooted at URI.

    Optionally only include paths with a certain file extension.

    Args:
        uri: The URI of a directory
        ext: The optional file extension to filter by
        fs: If supplied, use fs instead of automatically chosen
            :class:`.FileSystem` for ``uri``.
        **kwargs: Extra kwargs to pass to ``fs.list_paths()``.
    """
    if uri is None:
        return None

    if not fs:
        fs = get_file_system(uri)

    paths: list[str] = fs.ls(uri, detail=False, **kwargs)
    if ext != '':
        paths = [p for p in paths if p.endswith(ext)]
    return paths


def upload_or_copy(src_path: str, dst_uri: str,
                   fs: FileSystem | None = None) -> None:
    """Upload or copy a file.

    If ``dst_uri`` is local, the file is copied. Otherwise, it is uploaded.

    Args:
        src_path: Path to source file.
        dst_uri: URI of destination for file.
        fs: If supplied, use fs instead of automatically chosen
            :class:`.FileSystem` for ``dst_uri``.

    Raises:
        NotWritableError: if dst_uri cannot be written to
    """
    if not file_exists(src_path, fs=LocalFileSystem, include_dir=False):
        raise FileNotFoundError(f'{src_path} does not exist.')

    if is_local(dst_uri) and abspath(src_path) == abspath(dst_uri):
        return

    log.info(f'Uploading {src_path} to {dst_uri}')

    if not fs:
        fs = FileSystem.get_file_system(dst_uri, 'w')
    fs.copy_to(src_path, dst_uri)


def file_to_str(uri: str, fs: FileSystem | None = None) -> str:
    """Load contents of text file into a string.

    Args:
        uri: URI of file.
        fs: If supplied, use fs instead of automatically chosen FileSystem.

    Returns:
        Contents of text file.

    Raises:
        NotReadableError: If URI cannot be read.
    """
    with fsspec.open(uri, mode='r') as file:
        return file.


def str_to_file(content_str: str, uri: str,
                fs: FileSystem | None = None) -> None:
    """Writes string to text file.

    Args:
        content_str: string to write
        uri: URI of file to write
        fs: if supplied, use fs instead of automatically chosen FileSystem

    Raise:
        NotWritableError if uri cannot be written
    """
    if not fs:
        fs = get_file_system(uri)
    return fs.write_str(uri, content_str)


def file_to_json(uri: str) -> Any:
    """Load data from JSON file at uri."""
    with fsspec.open(uri, mode='r') as file:
        return json.load(file)


def json_to_file(obj: Any, uri: str) -> None:
    """Serialize obj to JSON and upload to uri."""
    with fsspec.open(uri, mode='w') as file:
        return json.dump(obj, file)


def zipdir(dir: str, zip_path: str) -> None:
    """Save a zip file with contents of directory.

    Contents of directory will be at root of zip file.

    Args:
        dir: directory to zip
        zip_path: path to zip file to create
    """
    make_dir(zip_path, use_dirname=True)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as ziph:
        with tqdm(desc='Zipping', delay=5) as bar:
            for dirpath, dirnames, filenames in os.walk(dir):
                for fn in filenames:
                    bar.set_postfix_str(fn)
                    src = join(dirpath, fn)
                    dst = join(dirpath[len(dir):], fn)
                    ziph.write(src, dst)
                    bar.update(1)


def unzip(zip_path: str, target_dir: str) -> None:
    """Unzip contents of zip file at zip_path into target_dir.

    Creates target_dir if needed.
    """
    make_dir(target_dir)
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(target_dir)


def is_local(uri: str) -> bool:
    return get_file_system(uri) == fsspec.filesystem('file')


def is_archive(uri: str) -> bool:
    """Check if the URI's extension represents an archived file."""
    formats = sum((fmts for _, fmts, _ in shutil.get_unpack_formats()), [])
    return any(uri.endswith(fmt) for fmt in formats)


def extract(uri: str,
            target_dir: str | None = None,
            download_dir: str | None = None) -> str:
    """Extract a compressed file."""
    if target_dir is None:
        target_dir = rv_config.get_cache_dir()
    make_dir(target_dir)
    local_path = download_if_needed(uri, download_dir)
    shutil.unpack_archive(local_path, target_dir)
    return target_dir


def get_tmp_dir() -> 'TemporaryDirectory':
    """Return temporary directory given by the :class:`RVConfig`.

    Returns:
        TemporaryDirectory: A context manager.
    """
    return rv_config.get_tmp_dir()


def uri_to_vsi_path(uri: str) -> str:
    """A function to convert URIs to VSI path strings.

    Args:
        uri: URI of the file. Acceptable URI schemes are file, s3, gs, http,
            https, and ftp.
    """
    URI_SCHEME_TO_VSI = {
        'http': 'vsicurl',
        'https': 'vsicurl',
        'ftp': 'vsicurl',
        's3': 'vsis3',
        'gs': 'vsigs',
    }
    parsed = urlparse(uri)
    scheme, netloc, path = parsed.scheme, parsed.netloc, parsed.path
    if scheme in URI_SCHEME_TO_VSI:
        return join('/', URI_SCHEME_TO_VSI[scheme], f'{netloc}{path}')
    # assume file schema
    return abspath(join(netloc, path))
