<?php
  session_start();
  include "connection.php";

  $name =  $_REQUEST["name"];
  $email = $_REQUEST["email"];
  $address =  $_REQUEST["address"];
  $phone = $_REQUEST["phone"];
  $username =  $_REQUEST["username"];
  $pass = $_REQUEST["pass"];


			$query = "INSERT INTO `user`(`name`, `username`, `pass`, `address`, `phone`, `email`) VALUES ('$name','$username','$pass','$address','$phone','$email')";

			
  if(mysqli_query($link, $query))
  {
			  $_SESSION['username'] = $username;
			  echo"<script> alert('Registration Successfull') </script>";
			  echo '<script>window.location.href = "index.html";</script>';
			  die();
   
	} else{
	    
	    	echo"<script> alert('Something wrong') </script>";
				echo '<script>window.location.href = "../index.html";</script>';
				die();
	}

?>