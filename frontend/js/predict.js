const token =
    localStorage.getItem("token");

if (!token) {

    window.location.href =
        "login.html";
}

const predictBtn =
    document.getElementById("predict-btn");

predictBtn.addEventListener(
    "click",
    predictSignal
);

async function predictSignal(event) {

    event.preventDefault();

    try {

        const token =
            localStorage.getItem("token");

        const koi_period =
            parseFloat(
                document.getElementById("koi_period").value
            );

        const koi_time0bk =
            parseFloat(
                document.getElementById("koi_time0bk").value
            );

        const koi_duration =
            parseFloat(
                document.getElementById("koi_duration").value
            );

        const koi_depth =
            parseFloat(
                document.getElementById("koi_depth").value
            );

        const koi_impact =
            parseFloat(
                document.getElementById("koi_impact").value
            );

        const koi_model_snr =
            parseFloat(
                document.getElementById("koi_model_snr").value
            );

        const koi_prad =
            parseFloat(
                document.getElementById("koi_prad").value
            );

        const koi_teq =
            parseFloat(
                document.getElementById("koi_teq").value
            );

        const koi_insol =
            parseFloat(
                document.getElementById("koi_insol").value
            );

        const koi_steff =
            parseFloat(
                document.getElementById("koi_steff").value
            );

        const response =
            await fetch(
                "https://kepler-s-planet-signal-review-system-production-e446.up.railway.app/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json",
                        "Authorization": `Bearer ${token}`
                    },

                    body: JSON.stringify({
                        koi_period,
                        koi_time0bk,
                        koi_duration,
                        koi_depth,
                        koi_impact,
                        koi_model_snr,
                        koi_prad,
                        koi_teq,
                        koi_insol,
                        koi_steff
                    })
                }
            );

        if (!response.ok) {

            const errorData =
                await response.json();

            document.getElementById(
                "result"
            ).innerHTML =

                `<p>Error: ${JSON.stringify(errorData)}</p>`;

            return;
        }

        const data =
            await response.json();

        document.getElementById(
            "result"
        ).innerHTML =

            `
            <h3>Prediction Result</h3>

            <p>
                <strong>Class:</strong>
                ${data.predicted_class}
            </p>

            <p>
                <strong>Confidence:</strong>
                ${data["Confidence Score"]}%
            </p>

            <p>
                <strong>Priority:</strong>
                ${data["Priority"]}
            </p>

            <p>
                <strong>Review Recommendation:</strong>
                ${data["Review Recommendation"]}
            </p>
            `;

    } catch (error) {

        console.error(error);

        document.getElementById(
            "result"
        ).innerHTML =

            `
            <p>
                Failed to connect to backend.
            </p>
            `;
    }
}