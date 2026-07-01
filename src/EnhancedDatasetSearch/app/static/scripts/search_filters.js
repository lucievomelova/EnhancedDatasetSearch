document.addEventListener('DOMContentLoaded', function() {
    initializeFilters();
});

function initializeFilters() {
    const filterInputs = document.querySelectorAll('.filter-input');

    // add event listeners to filter checkboxes
    filterInputs.forEach(input => {
        input.addEventListener('change', function() {
            applyFilters();
            updateSearchFormWithFilters();
        });
    });

    // add event listener to clear filters button
    const clearFiltersBtn = document.getElementById('clearFiltersBtn');
    if (clearFiltersBtn) {
        clearFiltersBtn.addEventListener('click', function(e) {
            e.preventDefault();
            clearAllFilters();
        });
    }
}

// apply filters to search results
function applyFilters() {
    const filterInputs = document.querySelectorAll('.filter-input:checked');
    const resultCards = document.querySelectorAll('.result-card');
    const noResultsFiltered = document.getElementById('noResultsFiltered');
    
    // organize selected filters by type
    const selectedFilters = {
        keywords: [],
        themes: [],
        categories: [],
        provider: [],
        temporal_coverage: [],
        spatial_coverage: []
    };
    
    filterInputs.forEach(input => {
        const filterType = input.getAttribute('data-filter-type');
        const value = input.value;
        if (selectedFilters[filterType]) {
            selectedFilters[filterType].push(value);
        }
    });
    
    // check if any filters are selected
    const hasActiveFilters = Object.values(selectedFilters).some(arr => arr.length > 0);
    let visibleCount = 0;
    
    resultCards.forEach(card => {
        const cardMatches = checkCardMatchesFilters(card, selectedFilters);
        
        if (cardMatches) {
            card.style.display = '';
            visibleCount++;
        } else {
            card.style.display = 'none';
        }
    });
    
    // show / hide no results message
    if (visibleCount === 0 && hasActiveFilters) {
        noResultsFiltered.style.display = '';
    } else {
        noResultsFiltered.style.display = 'none';
    }
    
    // update results count
    updateResultsCount(visibleCount);
}

// check if a result card matches the selected filters
function checkCardMatchesFilters(card, selectedFilters) {
    const cardData = {
        keywords: (card.getAttribute('data-keywords') || '').split(',').filter(v => v),
        themes: (card.getAttribute('data-themes') || '').split(',').filter(v => v),
        categories: (card.getAttribute('data-categories') || '').split(',').filter(v => v),
        provider: (card.getAttribute('data-provider') || '').trim(),
        temporal_coverage: (card.getAttribute('data-temporal') || '').split(',').filter(v => v),
        spatial_coverage: (card.getAttribute('data-spatial') || '').split(',').filter(v => v)
    };
    
    // check each filter type
    for (const [filterType, selectedValues] of Object.entries(selectedFilters)) {
        if (selectedValues.length === 0) {
            continue;  // no filter selected for this type -> skip
        }
        
        let cardHasValue = false;
        
        if (filterType === 'provider') {
            cardHasValue = selectedValues.includes(cardData.provider);  // for provider, do exact match
        } else {
            // for arrays, check if any selected value exists in card data
            cardHasValue = selectedValues.some(value => cardData[filterType].includes(value));
        }
        
        if (!cardHasValue) {
            return false;  // card doesn't match this filter -> it will be excluded
        }
    }

    return true;  // card matches all active filters
}


function clearAllFilters() {
    const filterInputs = document.querySelectorAll('.filter-input');
    filterInputs.forEach(input => {
        input.checked = false;
    });
    applyFilters();
    updateSearchFormWithFilters();
}


function updateResultsCount(count) {
    const resultsCountSpan = document.getElementById('resultsCount');
    if (resultsCountSpan) {
        resultsCountSpan.textContent = count;
    }
}


// Update search form with selected filters, this prepares filter data to be sent with next search
function updateSearchFormWithFilters() {
    const filterInputs = document.querySelectorAll('.filter-input:checked');
    const filtersData = {};
    
    filterInputs.forEach(input => {
        const filterType = input.getAttribute('data-filter-type');
        const value = input.value;
        
        if (!filtersData[filterType]) {
            filtersData[filterType] = [];
        }
        filtersData[filterType].push(value);
    });
    
    // Store filters in session storage for next search
    if (Object.keys(filtersData).length > 0) {
        sessionStorage.setItem('activeFilters', JSON.stringify(filtersData));
    } else {
        sessionStorage.removeItem('activeFilters');
    }
}

// Restore filters from session storage if they exist - call this on page load to restore previous filters
function restoreFiltersFromSession() {
    const activeFilters = sessionStorage.getItem('activeFilters');
    
    if (activeFilters) {
        try {
            const filtersData = JSON.parse(activeFilters);
            
            for (const [filterType, values] of Object.entries(filtersData)) {
                values.forEach(value => {
                    const input = document.querySelector(
                        `.filter-input[data-filter-type="${filterType}"][value="${value}"]`
                    );
                    if (input) {
                        input.checked = true;
                    }
                });
            }

            applyFilters();  // apply restored filters
        } catch (e) {
            console.error('Error restoring filters:', e);
        }
    }
}

window.addEventListener('load', function() {
    restoreFiltersFromSession();  // restore filters on page load
});

document.querySelectorAll(".filter-title").forEach(row => {
    row.addEventListener("click", () => {
        const toggle = row.querySelector(".filter-toggle");
        const targetId = toggle.dataset.target;
        const filterList = document.getElementById(targetId);

        if (!filterList) return;

        filterList.classList.toggle("open");

        toggle.textContent = filterList.classList.contains("open")
            ? "▲"
            : "▼";
    });
});