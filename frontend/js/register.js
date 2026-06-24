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

    const response =
        await fetch(
            "https://kepler-s-planet-signal-review-system-production-e446.up.railway.app//register",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    email: email,
                    password: password
                })
            }
        );

    const data =
        await response.json();

    document.getElementById("message")
        .textContent = data.message;

    console.log(data);
}