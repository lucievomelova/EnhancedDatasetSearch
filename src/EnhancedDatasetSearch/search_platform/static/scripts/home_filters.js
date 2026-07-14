document.querySelectorAll(".filter-toggle").forEach(button => {
    button.addEventListener("click", (e) => {
        e.stopPropagation();

        const targetId = button.dataset.target;
        const dropdown = document.getElementById(targetId);

        if (!dropdown) return;

        // close others (optional but recommended)
        document.querySelectorAll(".filter-dropdown").forEach(d => {
            if (d !== dropdown) d.classList.remove("open");
        });

        dropdown.classList.toggle("open");
    });
});

document.addEventListener("click", (e) => {
    if (!e.target.closest(".filter-item")) {
        document.querySelectorAll(".filter-dropdown").forEach(d => {
            d.classList.remove("open");
        });
    }
});
