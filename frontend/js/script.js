alert("Hello")
const testButton =
    document.getElementById("test-api");

testButton.addEventListener(
    "click",
    connectBackend
);

async function connectBackend() {

    const response =
        await fetch(
            "https://kepler-s-planet-signal-review-system-production-e446.up.railway.app//"
        );

    const data =
        await response.json();

    document.getElementById("backend-message").textContent = data.message;

}

console.log(
    document.getElementById("email").value
);