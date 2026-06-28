// reset chatbot memory on page load
window.addEventListener('load', async function() {
    try {
        await fetch('/chatbot/reset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
    } catch (error) {
        console.error('Error resetting chatbot:', error);
    }
});

document.getElementById('chatForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();
    if (!message) return;

    addMessage(message, 'user');  // add new message to chat
    userInput.value = '';
    
    const loadingMessageId = showLoadingDots();  // show loading dots in chat

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
        removeMessage(loadingMessageId);  // remove loading message

        if (data.success) {
            addMessage(data.response, 'assistant');
        } else {
            addMessage('Sorry, I encountered an error processing your request. Please try again.', 'assistant');
        }
    } catch (error) {
        console.error('Error:', error);
        removeMessage(loadingMessageId);
        addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
    }
});

const converter = new showdown.Converter();


function addMessage(text, sender, isLoading = false) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}-message`;
    
    // generate unique ID for this message
    const messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    messageDiv.id = messageId;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `${converter.makeHtml(escapeHtml(text))}`;

    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;  // scroll down
    return messageId;
}

function showLoadingDots() {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message assistant-message';
    
    // generate ID for the loading message
    const messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    messageDiv.id = messageId;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = 'Thinking<span class="loading-dots"></span>';

    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;  // scroll down    
    return messageId;
}

function removeMessage(messageId) {
    const messageElement = document.getElementById(messageId);
    if (messageElement) {
        messageElement.remove();
    }
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

document.getElementById('userInput').focus();  // focus on input when page loads
