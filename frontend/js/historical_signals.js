const token =
    localStorage.getItem("token");

if (!token) {

    window.location.href =
        "login.html";
}

const historyBtn =
    document.getElementById("history-btn");

historyBtn.addEventListener(
    "click",
    loadHistory
);

async function loadHistory() {

    try {

        const koi_period =
            parseFloat(
                document.getElementById(
                    "koi_period"
                ).value
            );

        const koi_depth =
            parseFloat(
                document.getElementById(
                    "koi_depth"
                ).value
            );

        const koi_prad =
            parseFloat(
                document.getElementById(
                    "koi_prad"
                ).value
            );

        const koi_steff =
            parseFloat(
                document.getElementById(
                    "koi_steff"
                ).value
            );

        const response =
            await fetch(
                "http://127.0.0.1:8000/history",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        koi_period,
                        koi_depth,
                        koi_prad,
                        koi_steff
                    })
                }
            );

        const data =
            await response.json();

        const planets =
            data["Historical Similar Planets"];

        let html =

            `
            <h2>
                Similar Historical Signals
            </h2>
            `;

        planets.forEach(
            planet => {

                html +=

                    `
                <div class="history-card">

                    <h3>
                        ${planet["Planet Name"]}
                    </h3>

                    <p>
                        <strong>
                            Disposition:
                        </strong>

                        ${planet["Disposition"]}
                    </p>

                    <p>
                        ${planet["Planet Summary"]}
                    </p>

                    <hr>

                </div>
                `;
            }
        );

        document.getElementById(
            "history-results"
        ).innerHTML = html;

    }

    catch (error) {

        console.error(error);

        document.getElementById(
            "history-results"
        ).innerHTML =

            `
            <p>
                Failed to retrieve
                historical signals.
            </p>
            `;
    }
}