const loginBtn =
    document.getElementById("login-btn");

loginBtn.addEventListener(
    "click",
    loginEvent
);

async function loginEvent(event) {

    event.preventDefault();

    const email =
        document.getElementById("email").value;

    const password =
        document.getElementById("password").value;

    const formData =
        new URLSearchParams();

    formData.append(
        "username",
        email
    );

    formData.append(
        "password",
        password
    );

    const response =
        await fetch(
            "http://127.0.0.1:8000/login",
            {
                method: "POST",
                body: formData
            }
        );

    const data =
        await response.json();

    console.log(data);

    localStorage.setItem(
        "token",
        data.access_token
    );

    window.location.href =
        "dashboard.html";

}