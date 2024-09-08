// script.js

// Fetch the CSV file (adjust the path as needed)
fetch('angry.csv')
    .then(response => response.text())
    .then(csvData => {
        const rows = csvData.split('\n');
        const tableBody = document.querySelector('#songTable tbody');

        rows.forEach(row => {
            const columns = row.split(',');
            if (columns.length === 5) {
                const [Name, Album, Artist, Link, Image] = columns;

                const newRow = document.createElement('tr');
                newRow.innerHTML = `
                    <td>${Name}</td>
                    <td>${Album}</td>
                    <td>${Artist}</td>
                    <td><a href="${Link}" target="_blank">Listen</a></td>
                    <td><img src="${Image}" alt="${Name}"></td>
                `;

                tableBody.appendChild(newRow);
            }
        });
    })
    .catch(error => console.error('Error fetching CSV:', error));
