from pathlib import Path
from werkzeug.utils import secure_filename
from services.logger import log
import zipfile
import tarfile
from datetime import datetime
import shutil
import py_compile
import subprocess
import sys
import json

BASE_DIR = Path(
    "/home/pi/tracker-mini-updater"
)

TRACKER_DIR = Path(
    "/home/pi/tracker-mini"
)

TEST_INSTALL_DIR = (
    BASE_DIR / "test-install"
)

CURRENT_FILE = (
    BASE_DIR / "current.json"
)

INSTALL_REQUEST_FILE = (
    BASE_DIR / "install-request.json"
)


UPLOAD_DIR = BASE_DIR / "uploads"
BACKUP_DIR = BASE_DIR / "backups"
STAGING_DIR = BASE_DIR / "staging"

MAX_PACKAGE_SIZE = (
    100 * 1024 * 1024
)


def save_package(file):

    filename = secure_filename(
        file.filename
    )

    save_path = (
        UPLOAD_DIR / filename
    )

    file.save(save_path)

    size = save_path.stat().st_size

    log(
        "UPDATE",
        "Package uploaded:",
        filename,
        f"({size} bytes)"
    )

    return {
        "success": True,
        "filename": filename,
        "size": size
    }


def list_backups():

    backups = []

    if not BACKUP_DIR.exists():
        return backups

    for item in sorted(
        BACKUP_DIR.iterdir(),
        reverse=True
    ):

        if item.is_dir():

            backups.append({
                "name": item.name
            })

    return backups


def validate_package(filename):

    package_path = (
        UPLOAD_DIR / filename
    )

    if not package_path.exists():

        log(
            "UPDATE",
            "Package not found:",
            filename
        )

        return {
            "success": False,
            "error": "Package not found"
        }

    try:

        with zipfile.ZipFile(
            package_path,
            "r"
        ) as z:

            names = z.namelist()

        has_backend = any(
            n.startswith("backend/")
            for n in names
        )

        has_frontend = any(
            n.startswith("frontend/")
            for n in names
        )

        package_type = "unknown"

        if has_backend and has_frontend:
            package_type = "full"

        elif has_backend:
            package_type = "backend"

        elif has_frontend:
            package_type = "frontend"

        log(
            "UPDATE",
            "Package validated:",
            filename,
            f"type={package_type}"
        )

        return {
            "success": True,
            "filename": filename,
            "type": package_type,
            "backend": has_backend,
            "frontend": has_frontend
        }

    except Exception as e:

        log(
            "UPDATE",
            "Validation failed:",
            str(e),
            level="ERROR"
        )

        return {
            "success": False,
            "error": str(e)
        }


def list_packages():

    packages = []

    if not UPLOAD_DIR.exists():
        return packages

    for item in sorted(
        UPLOAD_DIR.iterdir(),
        reverse=True
    ):

        if not item.is_file():
            continue

        if item.suffix.lower() != ".zip":
            continue

        info = validate_package(
            item.name
        )

        packages.append({
            "filename": item.name,
            "size": item.stat().st_size,
            "type": info.get(
                "type",
                "unknown"
            )
        })

    return packages


def create_backup():

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_dir = (
        BACKUP_DIR / timestamp
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    backend_src = (
        TRACKER_DIR / "backend"
    )

    frontend_src = (
        TRACKER_DIR / "frontend"
    )

    backend_tar = (
        backup_dir / "backend.tar.gz"
    )

    frontend_tar = (
        backup_dir / "frontend.tar.gz"
    )

    with tarfile.open(
        backend_tar,
        "w:gz"
    ) as tar:

        tar.add(
            backend_src,
            arcname="backend"
        )

    with tarfile.open(
        frontend_tar,
        "w:gz"
    ) as tar:

        tar.add(
            frontend_src,
            arcname="frontend"
        )

    log(
        "UPDATE",
        "Backup created:",
        timestamp
    )

    cleanup_old_backups()

    return {
        "success": True,
        "backup": timestamp
    }




def cleanup_old_backups():

    backups = sorted(
        [
            d
            for d in BACKUP_DIR.iterdir()
            if d.is_dir()
        ]
    )

    while len(backups) > 3:

        oldest = backups.pop(0)

        shutil.rmtree(
            oldest,
            ignore_errors=True
        )

        log(
            "UPDATE",
            "Old backup removed:",
            oldest.name
        )


def extract_package(filename):

    package_path = (
        UPLOAD_DIR / filename
    )

    if not package_path.exists():

        return {
            "success": False,
            "error": "Package not found"
        }

    try:

        if STAGING_DIR.exists():

            shutil.rmtree(
                STAGING_DIR
            )

        STAGING_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        with zipfile.ZipFile(
            package_path,
            "r"
        ) as z:

            z.extractall(
                STAGING_DIR
            )

        contents = []

        if (STAGING_DIR / "backend").exists():
            contents.append(
                "backend"
            )

        if (STAGING_DIR / "frontend").exists():
            contents.append(
                "frontend"
            )

        log(
            "UPDATE",
            "Package extracted:",
            filename,
            f"contents={contents}"
        )

        return {
            "success": True,
            "filename": filename,
            "contents": contents
        }

    except Exception as e:

        log(
            "UPDATE",
            "Extraction failed:",
            str(e),
            level="ERROR"
        )

        return {
            "success": False,
            "error": str(e)
        }


def test_backend_syntax():

    backend_dir = (
        STAGING_DIR / "backend"
    )

    if not backend_dir.exists():

        return {
            "success": False,
            "error": "Backend directory not found"
        }

    checked = 0
    errors = []

    for py_file in backend_dir.rglob(
        "*.py"
    ):

        try:

            py_compile.compile(
                str(py_file),
                doraise=True
            )

            checked += 1

        except Exception as e:

            errors.append({
                "file": str(
                    py_file.relative_to(
                        backend_dir
                    )
                ),
                "error": str(e)
            })

    if errors:

        log(
            "UPDATE",
            f"Python syntax test FAILED ({len(errors)} errors)",
            level="ERROR"
        )

        return {
            "success": False,
            "checked": checked,
            "errors": errors
        }

    log(
        "UPDATE",
        f"Python syntax test OK ({checked} files)"
    )

    return {
        "success": True,
        "checked": checked,
        "errors": []
    }


def test_package(filename):

    syntax = {
        "checked": 0
    }

    log(
        "UPDATE",
        "Testing package:",
        filename
    )

    validation = validate_package(
        filename
    )

    

    if not validation["success"]:

        return {
            "success": False,
            "stage": "validation",
            "error": validation.get(
                "error"
            )
        }

    extraction = extract_package(
        filename
    )

    if not extraction["success"]:

        return {
            "success": False,
            "stage": "extraction",
            "error": extraction.get(
                "error"
            )
        }

    if validation.get(
            "backend",
            False
        ):

            backend_structure = (
                test_backend_structure()
            )

            if not backend_structure[
                "success"
            ]:

                return {
                    "success": False,
                    "stage": "backend_structure",
                    "missing": backend_structure[
                        "missing"
                    ]
                }

    if validation.get(
        "frontend",
        False
    ):

        frontend_structure = (
            test_frontend_structure()
        )

        if not frontend_structure[
            "success"
        ]:

            return {
                "success": False,
                "stage": "frontend_structure",
                "missing": frontend_structure[
                    "missing"
                ]
            }



    if validation.get(
        "backend",
        False
    ):

        syntax = test_backend_syntax()

        if not syntax["success"]:

            return {
                "success": False,
                "stage": "syntax",
                "errors": syntax[
                    "errors"
                ]
            }

    log(
        "UPDATE",
        "Package test PASSED:",
        filename
    )

    return {
        "success": True,
        "package": filename,
        "type": validation[
            "type"
        ],
        "backend": validation[
            "backend"
        ],
        "frontend": validation[
            "frontend"
        ],
        "checked_files": syntax.get(
            "checked",
            0
        ) if validation.get(
            "backend",
            False
        ) else 0
    }


def test_backend_structure():

    backend_dir = (
        STAGING_DIR / "backend"
    )

    required = [
        backend_dir / "app.py",
        backend_dir / "routes",
        backend_dir / "services"
    ]

    missing = []

    for item in required:

        if not item.exists():

            missing.append(
                str(
                    item.relative_to(
                        backend_dir
                    )
                )
            )

    if missing:

        log(
            "UPDATE",
            "Backend structure FAILED",
            str(missing),
            level="ERROR"
        )

        return {
            "success": False,
            "missing": missing
        }

    log(
        "UPDATE",
        "Backend structure OK"
    )

    return {
        "success": True
    }


def test_frontend_structure():

    frontend_dir = (
        STAGING_DIR / "frontend"
    )

    required = [
        frontend_dir / "index.html",
        frontend_dir / "css",
        frontend_dir / "js"
    ]

    missing = []

    for item in required:

        if not item.exists():

            missing.append(
                str(
                    item.relative_to(
                        frontend_dir
                    )
                )
            )

    if missing:

        log(
            "UPDATE",
            "Frontend structure FAILED",
            str(missing),
            level="ERROR"
        )

        return {
            "success": False,
            "missing": missing
        }

    log(
        "UPDATE",
        "Frontend structure OK"
    )

    return {
        "success": True
    }


def test_install():

    try:

        if TEST_INSTALL_DIR.exists():

            shutil.rmtree(
                TEST_INSTALL_DIR
            )

        TEST_INSTALL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        backend_files = 0
        frontend_files = 0

        if (
            STAGING_DIR / "backend"
        ).exists():

            shutil.copytree(
                STAGING_DIR / "backend",
                TEST_INSTALL_DIR / "backend"
            )

            backend_files = len(
                list(
                    (
                        TEST_INSTALL_DIR
                        / "backend"
                    ).rglob("*")
                )
            )

        if (
            STAGING_DIR / "frontend"
        ).exists():

            shutil.copytree(
                STAGING_DIR / "frontend",
                TEST_INSTALL_DIR / "frontend"
            )

            frontend_files = len(
                list(
                    (
                        TEST_INSTALL_DIR
                        / "frontend"
                    ).rglob("*")
                )
            )

        if (
            TRACKER_DIR / "config"
        ).exists():

            shutil.copytree(
                TRACKER_DIR / "config",
                TEST_INSTALL_DIR / "config"
            )

            log(
                "UPDATE",
                "Config copied from live installation"
            )


        log(
            "UPDATE",
            "Test install OK"
        )

        return {
            "success": True,
            "backend_files": backend_files,
            "frontend_files": frontend_files
        }

    except Exception as e:

        log(
            "UPDATE",
            "Test install FAILED",
            str(e),
            level="ERROR"
        )

        return {
            "success": False,
            "error": str(e)
        }


def test_backend_import():

    backend_dir = (
        TEST_INSTALL_DIR / "backend"
    )

    if not backend_dir.exists():

        return {
            "success": False,
            "error": "Test install backend not found"
        }

    try:

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import app"
            ],
            cwd=str(
                backend_dir
            ),
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:

            log(
                "UPDATE",
                "Backend import FAILED",
                result.stderr,
                level="ERROR"
            )

            return {
                "success": False,
                "error": result.stderr
            }

        log(
            "UPDATE",
            "Backend import OK"
        )

        return {
            "success": True
        }

    except Exception as e:

        log(
            "UPDATE",
            "Backend import FAILED",
            str(e),
            level="ERROR"
        )

        return {
            "success": False,
            "error": str(e)
        }


def read_current():

    if not CURRENT_FILE.exists():

        return {
            "installed_package": None,
            "installed_at": None,
            "backup": None,
            "status": "unknown"
        }

    with open(
        CURRENT_FILE,
        "r"
    ) as f:

        return json.load(f)

def write_current(data):

    with open(
        CURRENT_FILE,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )



def _restore_to_directory(
    backup_name,
    target_dir
):

    backup_dir = (
        BACKUP_DIR / backup_name
    )

    if not backup_dir.exists():

        return {
            "success": False,
            "error": "Backup not found"
        }

    backend_tar = (
        backup_dir / "backend.tar.gz"
    )

    frontend_tar = (
        backup_dir / "frontend.tar.gz"
    )

    try:

        if (
            target_dir / "backend"
        ).exists():

            shutil.rmtree(
                target_dir / "backend"
            )

        if (
            target_dir / "frontend"
        ).exists():

            shutil.rmtree(
                target_dir / "frontend"
            )

        with tarfile.open(
            backend_tar,
            "r:gz"
        ) as tar:

            tar.extractall(
                target_dir
            )

        with tarfile.open(
            frontend_tar,
            "r:gz"
        ) as tar:

            tar.extractall(
                target_dir
            )

        return {
            "success": True
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }



def restore_backup_test(
    backup_name
):

    result = _restore_to_directory(
        backup_name,
        TEST_INSTALL_DIR
    )

    if result["success"]:

        log(
            "UPDATE",
            "Test restore OK:",
            backup_name
        )

    return result




def restore_backup(
    backup_name
):

    result = _restore_to_directory(
        backup_name,
        TRACKER_DIR
    )

    if result["success"]:

        log(
            "UPDATE",
            "Backup restored:",
            backup_name
        )

    return result



def pre_install_check(
    filename
):

    package_test = test_package(
        filename
    )

    if not package_test[
        "success"
    ]:

        return {
            "success": False,
            "stage": "package_test",
            "details": package_test
        }

    install_test = (
        test_install()
    )

    if not install_test[
        "success"
    ]:

        return {
            "success": False,
            "stage": "test_install",
            "details": install_test
        }

    import_test = (
        test_backend_import()
    )

    if not import_test[
        "success"
    ]:

        return {
            "success": False,
            "stage": "test_import",
            "details": import_test
        }

    backup = create_backup()

    if not backup[
        "success"
    ]:

        return {
            "success": False,
            "stage": "backup",
            "details": backup
        }

    log(
        "UPDATE",
        "Pre-install check PASSED"
    )

    return {
        "success": True,
        "ready": True,
        "backup": backup[
            "backup"
        ],
        "steps": [

            "Package validated",

            "Package extracted",

            "Backend structure verified",

            "Frontend structure verified",

            "Python syntax verified",

            "Test installation verified",

            "Backend import verified",

            "Backup created"

        ]
    }


def create_install_request(
    filename,
    backup_name=None
):

    try:

        request_data = {

            "package": filename,

            "backup": backup_name,

            "created_at":
                datetime.now().isoformat(),

            "status": "pending"

        }

        with open(
            INSTALL_REQUEST_FILE,
            "w"
        ) as f:

            json.dump(
                request_data,
                f,
                indent=4
            )

        log(
            "UPDATE",
            "Install request created:",
            filename
        )

        return {

            "success": True,

            "package": filename,

            "backup": backup_name,

            "status": "pending"

        }

    except Exception as e:

        log(
            "UPDATE",
            "Install request FAILED",
            str(e),
            level="ERROR"
        )

        return {

            "success": False,

            "error": str(e)

        }