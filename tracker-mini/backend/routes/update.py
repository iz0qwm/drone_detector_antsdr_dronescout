from flask import Blueprint, jsonify, request
from services.logger import log
from services.updater import (
    save_package,
    list_backups,
    validate_package,
    list_packages,
    create_backup,
    extract_package,
    test_backend_syntax,
    test_package,
    test_install,
    test_backend_import,
    read_current,
    restore_backup,
    restore_backup_test,
    pre_install_check,
    create_install_request
)



update_bp = Blueprint(
    "update",
    __name__
)

log(
    "UPDATE",
    "Blueprint loaded"
)



@update_bp.route(
    "/status",
    methods=["GET"]
)
def update_status():

    log(
        "UPDATE",
        "Status request"
    )

    return jsonify({
        "service": "updater",
        "status": "ok"
    })




@update_bp.route(
    "/upload",
    methods=["POST"]
)
def upload_package():

    if "file" not in request.files:

        log(
            "UPDATE",
            "Upload request without file"
        )

        return jsonify({
            "success": False,
            "error": "No file"
        }), 400

    file = request.files["file"]

    if file.filename == "":

        log(
            "UPDATE",
            "Upload request with empty filename"
        )

        return jsonify({
            "success": False,
            "error": "Empty filename"
        }), 400

    return jsonify(
        save_package(file)
    )



@update_bp.route(
    "/backups",
    methods=["GET"]
)
def get_backups():

    return jsonify(
        list_backups()
    )


@update_bp.route(
    "/package/<filename>",
    methods=["GET"]
)
def package_info(filename):

    return jsonify(
        validate_package(filename)
    )


@update_bp.route(
    "/packages",
    methods=["GET"]
)
def get_packages():

    return jsonify(
        list_packages()
    )


@update_bp.route(
    "/backup",
    methods=["POST"]
)
def backup():

    return jsonify(
        create_backup()
    )


@update_bp.route(
    "/extract/<filename>",
    methods=["POST"]
)
def extract(filename):

    return jsonify(
        extract_package(
            filename
        )
    )


@update_bp.route(
    "/test/backend",
    methods=["POST"]
)
def test_backend():

    return jsonify(
        test_backend_syntax()
    )

@update_bp.route(
    "/test/<filename>",
    methods=["POST"]
)
def test(filename):

    return jsonify(
        test_package(
            filename
        )
    )


@update_bp.route(
    "/test-install",
    methods=["POST"]
)
def run_test_install():

    return jsonify(
        test_install()
    )


@update_bp.route(
    "/test-import",
    methods=["POST"]
)
def test_import():

    return jsonify(
        test_backend_import()
    )


@update_bp.route(
    "/current",
    methods=["GET"]
)
def current():

    return jsonify(
        read_current()
    )


@update_bp.route(
    "/restore/<backup_name>",
    methods=["POST"]
)
def restore(backup_name):

    return jsonify(
        restore_backup(
            backup_name
        )
    )


@update_bp.route(
    "/restore-test/<backup_name>",
    methods=["POST"]
)
def restore_test(
    backup_name
):

    return jsonify(
        restore_backup_test(
            backup_name
        )
    )


@update_bp.route(
    "/pre-install/<filename>",
    methods=["POST"]
)
def pre_install(
    filename
):

    return jsonify(
        pre_install_check(
            filename
        )
    )


@update_bp.route(
    "/request-install/<filename>",
    methods=["POST"]
)
def request_install(filename):

    backup = None

    data = request.get_json(
        silent=True
    )

    if data:

        backup = data.get(
            "backup"
        )

    return jsonify(
        create_install_request(
            filename,
            backup
        )
    )