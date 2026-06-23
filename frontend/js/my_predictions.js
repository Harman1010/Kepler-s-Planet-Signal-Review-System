const token =
    localStorage.getItem("token");

if (!token) {

    window.location.href =
        "login.html";
}

window.onload =
    loadPredictions;

async function loadPredictions() {

    try {

        const token =
            localStorage.getItem("token");

        const response =
            await fetch(
                "http://127.0.0.1:8000/predictions",
                {
                    headers: {
                        "Authorization":
                            `Bearer ${token}`
                    }
                }
            );

        const predictions =
            await response.json();

        let rows =

            `
            <tr>
                <th>ID</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Priority</th>
                <th>Status</th>
            </tr>
            `;

        if (predictions.length === 0) {

            rows +=

                `
                <tr>
                    <td colspan="5">
                        No predictions found
                    </td>
                </tr>
                `;
        }

        predictions.forEach(
            prediction => {

                rows += `
<tr>

    <td>${prediction.id}</td>

    <td>${prediction.prediction}</td>

    <td>${prediction.confidence}%</td>

    <td>${prediction.priority}</td>

    <td>${prediction.review_status}</td>

    <td>

        <select id="status-${prediction.id}">

            <option value="Immediate Review">
                Immediate Review
            </option>

            <option value="Scientist Validation">
                Scientist Validation
            </option>

            <option value="Review Queue">
                Review Queue
            </option>

            <option value="Auto-Filtered">
                Auto-Filtered
            </option>

        </select>

        <button
            onclick="updateStatus(${prediction.id})">
            Update
        </button>

    </td>

    <td>

        <button
            onclick="deletePrediction(${prediction.id})">
            Delete
        </button>

    </td>

</tr>
`;
            }
        );

        document.getElementById(
            "prediction-table"
        ).innerHTML = rows;

    }

    catch (error) {

        console.error(error);

        document.getElementById(
            "prediction-table"
        ).innerHTML =

            `
            <tr>
                <td colspan="5">
                    Failed to load predictions
                </td>
            </tr>
            `;
    }
}

async function updateStatus(id) {

    const token =
        localStorage.getItem("token");

    const status =
        document.getElementById(
            `status-${id}`
        ).value;

    await fetch(
        `http://127.0.0.1:8000/predictions/${id}`,
        {
            method: "PATCH",

            headers: {
                "Content-Type":
                    "application/json",

                "Authorization":
                    `Bearer ${token}`
            },

            body: JSON.stringify({
                review_status: status
            })
        }
    );

    loadPredictions();
}

async function deletePrediction(id) {

    const token =
        localStorage.getItem("token");

    await fetch(
        `http://127.0.0.1:8000/predictions/${id}`,
        {
            method: "DELETE",

            headers: {
                "Authorization":
                    `Bearer ${token}`
            }
        }
    );

    loadPredictions();
}