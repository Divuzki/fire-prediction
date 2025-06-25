
const predictButton = document.getElementById('predict-button');
if (predictButton != null){
		let selected_img = null;

		document.addEventListener('DOMContentLoaded', function() {
		    const imageInput = document.getElementById('image-input');
		    const selectedImg = document.getElementById('selected-img');

		    imageInput.addEventListener('change', function() {
		        const file = this.files[0];
						document.getElementById('prediction-result').textContent = '';
		        if (file) {
		            const reader = new FileReader();
		            reader.onload = function(event) {
										selected_img = event.target.result;
		                selectedImg.src = selected_img;
										uploadImage(file);
		            };
		            reader.readAsDataURL(file);
		        }else{
							console.log('No Image Selected');
						}
		    });
		});

		const predictionResult = document.getElementById('prediction-result');
		predictButton.addEventListener('click', function() {
			fetch('/make_predictions', {
					method: 'POST',
					headers: {
							'Content-Type': 'application/json'
					},
			})
			.then(response => response.json())
			.then(data => {
					if (data.error) {
							predictionResult.textContent = 'Error: ' + data.error;
					} else {
							predictionResult.textContent = '';
							predictionResult.innerHTML += "<p><b>Predicted Meal:</b> " + data.meal + "</p>";
							predictionResult.innerHTML += "<p><b>Ingredients:</b> " + data.ingredients + "</p>";
							predictionResult.innerHTML += "<p><b>Recommendation:</b> " + data.recommendation + "</p>";
					}
			})
			.catch(error => {
					predictionResult.textContent = 'An error occurred while making predictions.';
			});
		});

		async function uploadImage(file) {
	        if (!file) {
	          document.getElementById('message').innerText = 'No file selected';
	          return;
	        }

	        const formData = new FormData();
	        formData.append('file', file);

	        try {
	          const response = await fetch('/upload', {
	            method: 'POST',
	            body: formData
	          });

	          const result = await response.json();
	          console.log(result);
	        } catch (error) {
	          console.log(error);
	        }
	      }

}


//TRAIN MODELS
var train_button = document.getElementById('train-button');
if (train_button != null){
	const trainStatus = document.getElementById('train-status');

		document.addEventListener('DOMContentLoaded', function() {
	    const trainButton = document.getElementById('train-button');

	    trainButton.addEventListener('click', function() {
			trainStatus.textContent = 'Training model...';

			fetch('/train', {
			    method: 'POST'
			})
			.then(response => response.json())
			.then(data => {

			    if (data.message) {
			        trainStatus.textContent = data.message;
			    } else if (data.error) {
			        trainStatus.textContent = data.error;
			    }
					if (data.results) {
	            trainStatus.innerHTML += '<h3>Results:</h3>';
	            data.results.forEach(result => {
	                trainStatus.innerHTML += `<p>${result}</p>`;
	            });
	        }
					if (data.best_values) {
	            trainStatus.innerHTML += '<h3>Best Values:<h3>';
	            data.best_values.forEach(value => {
	                trainStatus.innerHTML += `<p>${value}</p>`;
	            });
	        }
			})
			.catch(error => {
			    trainStatus.textContent = 'An error occurred while training the model.' + error;
			});
		});

		document.getElementById('id_btn_train_clear').addEventListener('click', ()=>{
			trainStatus.innerHTML = '';
		});
	});
}



//USER
var userTable = document.getElementById('userTable');
if (userTable != null){

	function populateUserTable(users) {
    const tableBody = document.querySelector('#userTable tbody');
    tableBody.innerHTML = ''; // Clear existing table rows

    users.forEach(user => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${user.email}</td>
            <td>${user.username}</td>
            <td>${user.name}</td>
            <td>${user.phone}</td>
            <td>${user.department}</td>
        `;
        tableBody.appendChild(row);
    });
	}

	function fetchUserDetails() {
		let id_user_email = document.getElementById('id_user_email').value;
	    fetch('/user-details?email=' + id_user_email) // Change the route to match your Flask route
	        .then(response => response.json())
	        .then(data => {
	            if (data.success) {
	                populateUserTable(data.users);
	            } else {
	                console.error('Failed to fetch user details:', data.error);
	            }
	        })
	        .catch(error => {
	            console.error('Error fetching user details:', error);
	        });
	}

	fetchUserDetails();

}
