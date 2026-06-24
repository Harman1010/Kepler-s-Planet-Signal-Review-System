const registerBtn =
    document.getElementById("register-btn");

registerBtn.addEventListener(
    "click",
    registerUser
);

async function registerUser(event) {

    event.preventDefault();

    const email =
        document.getElementById("email").value;

    const password =
        document.getElementById("password").value;

    try {

        const response =
            await fetch(
                "https://kepler-s-planet-signal-review-system-production-e446.up.railway.app/register",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        email: email,
                        password: password
                    })
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            document.getElementById(
                "message"
            ).textContent =
                data.detail ||
                "Registration failed";

            return;
        }

        document.getElementById(
            "message"
        ).textContent =
            data.message;

        console.log(data);

    }

    catch (error) {

        console.error(error);

        document.getElementById(
            "message"
        ).textContent =
            "Unable to connect to server";
    }
}