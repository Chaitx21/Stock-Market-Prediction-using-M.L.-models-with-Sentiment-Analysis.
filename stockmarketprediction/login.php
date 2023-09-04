<!DOCTYPE html>
<html>
  <head>
    <title>Stock Market Prediction</title>
    <style>
      body {
        background-image: url("login.jpg");
        background-size: cover;
        background-position: center;
        font-family: Arial, sans-serif;
        color: #333;
		margin-left: -900px;
      }
      form {
        background-color: #fff;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
        max-width: 300px;
        margin: 100px auto;
      }
      label {
        display: block;
        margin-bottom: 8px;
      }
      input[type="text"],
      input[type="password"] {
        display: block;
        width: 90%;
        padding: 10px;
        font-size: 16px;
        border: 1px solid #ccc;
        border-radius: 4px;
        margin-bottom: 20px;
      }
      button[type="submit"] {
        background-color: #333;
        color: #fff;
        padding: 10px 20px;
        border: none;
        border-radius: 4px;
        font-size: 16px;
        cursor: pointer;
      }
    </style>
  </head>
  <body>
  <br>
  <br>
  <br>
  <br>
  <br>
  <br>
  <br>
  <br>
  <br>
    <form action="login_back.php" method="POST">
	<h1>LOG-IN</h1>
      <label for="username">Username</label>
      <input type="text" id="username" name="username" required>
      <label for="password">Password</label>
      <input type="password" id="password" name="pass" required>
      <button type="submit">Sign In</button>
    </form>
  </body>
</html>
