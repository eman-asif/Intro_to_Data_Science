// JavaScript for the Solar Energy Solutions Webpage

// Function to display FAQ answers
function toggleAnswer(answerId) {
    var answer = document.getElementById(answerId);
    if (answer.style.display === "block") {
        answer.style.display = "none";
    } else {
        answer.style.display = "block";
    }
}

// Function to handle form submission (basic validation)
function submitForm() {
    var name = document.getElementById("name").value;
    var email = document.getElementById("email").value;
    var message = document.getElementById("message").value;

    if (name === "" || email === "" || message === "") {
        alert("Please fill in all fields before submitting.");
    } else {
        alert("Thank you, " + name + "! We have received your message.");
        // Reset form
        document.getElementById("contactForm").reset();
    }
}

// Event listener for the form submit button
document.getElementById("submitBtn").addEventListener("click", function(event) {
    event.preventDefault();  // Prevent form from refreshing the page
    submitForm();
});
