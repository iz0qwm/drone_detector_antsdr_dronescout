window.DRONES = window.DRONES || {};

DRONES.fetchRemoteIdAircraft = async function () {

    try {

        const response = await fetch(
            "/api/remoteid/aircraft"
        );

        if (!response.ok) {
            return [];
        }

        return await response.json();

        const data =
            await response.json();

        console.log(
            "[DRONES]",
            data.length,
            "aircraft"
        );

        return data;

    } catch (err) {

        console.error(
            "[DRONES]",
            err
        );

        return [];
    }
};