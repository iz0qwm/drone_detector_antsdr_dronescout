let currentPackage = null;
let currentBackup = null;

document.addEventListener(
    "DOMContentLoaded",
    () => {

        document
            .getElementById(
                "uploadUpdateBtn"
            )
            ?.addEventListener(
                "click",
                uploadAndVerifyPackage
            );

        document
            .getElementById(
                "installUpdateBtn"
            )
            ?.addEventListener(
                "click",
                createInstallRequest
            );

        loadCurrentVersion();
    }
);

async function uploadAndVerifyPackage() {

    const file =
        document.getElementById(
            "updateFile"
        ).files[0];

    if (!file) {

        alert(
            "Select a ZIP package"
        );

        return;
    }

    const status =
        document.getElementById(
            "updateStatus"
        );

    status.innerHTML =
        "Uploading package...";

    const form =
        new FormData();

    form.append(
        "file",
        file
    );

    try {

        const uploadRes =
            await fetch(
                "/api/update/upload",
                {
                    method: "POST",
                    body: form
                }
            );

        const upload =
            await uploadRes.json();

        currentPackage =
            upload.filename;

        status.innerHTML =
            "Verifying package...";

        const verifyRes =
            await fetch(
                `/api/update/pre-install/${currentPackage}`,
                {
                    method: "POST"
                }
            );

        const verify =
            await verifyRes.json();

        if (
            !verify.success
        ) {

            let details = "";

            if (verify.details?.error) {
                details = `<pre>${verify.details.error}</pre>`;
            }
            else if (verify.details?.errors) {
                details =
                    `<pre>${JSON.stringify(
                        verify.details.errors,
                        null,
                        2
                    )}</pre>`;
            }
            else if (verify.details?.missing) {
                details =
                    `<pre>${verify.details.missing.join("\n")}</pre>`;
            }

            status.innerHTML = `
                <b>Verification failed</b><br>
                Stage: ${verify.stage}
                <br><br>
                ${details}
            `;

            return;
        }

        currentBackup =
            verify.backup;

        let stepsHtml = "";

        if (verify.steps) {

            verify.steps.forEach(step => {

                stepsHtml += `
                    ✓ ${step}<br>
                `;

            });

        }

        status.innerHTML = `

            <div class="update-banner update-banner-ready">
                SYSTEM READY FOR UPDATE
            </div>

            ${stepsHtml}

            <hr>

            <b>READY TO INSTALL</b>
            <br><br>
            Package:
            ${currentPackage}
            <br>
            Backup:
            ${currentBackup}
        `;

        document
            .getElementById(
                "uploadUpdateBtn"
            )
            .style.display =
                "none";

        document
            .getElementById(
                "installUpdateBtn"
            )
            .style.display =
                "block";

    } catch(err) {

        console.error(err);

        status.innerHTML =
            "Update failed";

    }
}



async function createInstallRequest() {

    const installBtn =
        document.getElementById(
            "installUpdateBtn"
        );

    installBtn.disabled = true;

    const status =
        document.getElementById(
            "updateStatus"
        );

    status.innerHTML =
        "Creating install request...";

    const res =
        await fetch(
            `/api/update/request-install/${currentPackage}`,
            {
                method: "POST",
                headers: {
                    "Content-Type":
                        "application/json"
                },
                body: JSON.stringify({
                    backup:
                        currentBackup
                })
            }
        );

    const data =
        await res.json();

    if (!data.success) {

        status.innerHTML =
            `<b>Request failed</b><br>
            ${data.error || "Unknown error"}`;

        installBtn.disabled = false;
        return;
    }

    document
        .getElementById(
            "installUpdateBtn"
        )
        .style.display =
            "none";

        let countdown = 50;

        const refreshTimer =
            setInterval(() => {

                countdown--;

                const timerEl =
                    document.getElementById(
                        "updateCountdown"
                    );

                if (timerEl) {

                    timerEl.textContent =
                        countdown;
                }

                if (countdown <= 0) {

                    clearInterval(
                        refreshTimer
                    );

                    window.location.reload();

                }

            }, 1000);

            
    status.innerHTML = `

        <div class="update-banner update-banner-installing">
            INSTALL REQUEST CREATED
        </div>

        Package:
        ${data.package}

        <br>

        Backup:
        ${data.backup}

        <hr>

        The update package has been queued.

        <br><br>

        The system will automatically:

        <br>

        ✓ Stop services

        <br>

        ✓ Install the new software

        <br>

        ✓ Verify correct startup

        <br>

        ✓ Restore backup if necessary

        <br><br>

        This page may become temporarily unavailable during the update.

        <br><br>

        Page refresh in
        <span id="updateCountdown">
        50
        </span>
        seconds...
    `;
}


async function loadCurrentVersion() {

    try {

        const res =
            await fetch(
                "/api/update/current"
            );

        const data =
            await res.json();

        const rollbackInfo =
            data.rollback_reason
                ? `<br>
                Reason:
                ${data.rollback_reason}`
                : "";

        document
            .getElementById(
                "currentVersionInfo"
            )
            .innerHTML = `

                <b>Installed Version</b>

                <br><br>

                Package:
                ${data.installed_package || "Unknown"}

                <br>

                Installed:
                ${data.installed_at || "Unknown"}

                <br>

                Status:
                ${data.status || "Unknown"}

                ${rollbackInfo}

            `;

    } catch(e) {

        console.error(e);

    }
}