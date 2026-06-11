from config import (
    SETTINGS,
    save_settings
)

def get_dsc_settings():

    return SETTINGS.get(
        "dsc",
        {}
    )


def update_dsc_settings(data):

    SETTINGS["dsc"] = {

        "node_name":
            data.get(
                "node_name",
                "DSC Node"
            ),

        "position_source":
            data.get(
                "position_source",
                "manual"
            ),

        "lat":
            data.get("lat"),

        "lon":
            data.get("lon")
    }

    save_settings()

    return SETTINGS["dsc"]