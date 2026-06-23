const logoutBtn =
    document.getElementById("logout-btn");

logoutBtn.addEventListener(
    "click",
    logout
);

function logout() {

    localStorage.removeItem(
        "token"
    );

    window.location.href =
        "login.html";
}