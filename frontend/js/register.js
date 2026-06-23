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
            "http://127.0.0.1:8000/register",
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