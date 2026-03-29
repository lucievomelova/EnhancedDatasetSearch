document.getElementById('chatForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();

    if (!message) return;

    addMessage(message, 'user');  // add new message to chat
    userInput.value = '';

    // send message to backend
    try {
        const response = await fetch('/chatbot/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: message })
        });

        const data = await response.json();

        if (data.success) {
            addMessage(data.response, 'assistant');
        } else {
            addMessage('Sorry, I encountered an error processing your request. Please try again.', 'assistant');
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
    }
});

const converter = new showdown.Converter();


function addMessage(text, sender) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `${converter.makeHtml(escapeHtml(text))}`;

    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);

    messagesDiv.scrollTop = messagesDiv.scrollHeight;  // scroll down
}


function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Focus on input when page loads
document.getElementById('userInput').focus();