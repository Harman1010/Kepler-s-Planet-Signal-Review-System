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
            "http://127.0.0.1:8000/"
        );

    const data =
        await response.json();

    document.getElementById("backend-message").textContent = data.message;

}

console.log(
    document.getElementById("email").value
);