<!DOCTYPE html>
<html lang="en">

<head>
	<meta charset="UTF-8">
	<meta http-equiv="X-UA-Compatible" content="IE=edge">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<meta name="description"
		content="Responsive Bootstrap4 Shop Template, Created by Imran Hossain from https://imransdesign.com/">

	<!-- title -->
	<title>Contact</title>

<jsp:include page="styles.jsp" />
	

</head>

<body>

	<!--PreLoader-->
	<div class="loader">
		<div class="loader-inner">
			<div class="circle"></div>
		</div>
	</div>
	<!--PreLoader Ends-->

<!-- header -->
	<div class="sticky-wrapper is-sticky" style="background-color: #051922" id="sticker">
		<div class="container">
			<div class="row">
				<div class="col-lg-12 col-sm-12 text-center">
					<div class="main-menu-wrap">
						<!-- logo -->
						<div class="site-logo">
							<a href="index.html">
								<img src="assets/img/newlogo.png" alt="">
							</a>
						</div>
						<!-- logo -->

						<!-- menu start -->
						<nav class="main-menu">
							<ul>
								<li class="current-list-item" ><a href="login.jsp">Login</a>
								<a href="signup.jsp">Sign Up</a>
								</li>


								

									
								
							</ul>
						</nav>
						<a class="mobile-show search-bar-icon" href="#"><i class="fas fa-search"></i></a>
						<div class="mobile-menu"></div>
						<!-- menu end -->
					</div>
				</div>
			</div>
		</div>
	</div>
	<!-- end header -->

	<!-- search area -->
	<div class="search-area">
		<div class="container">
			<div class="row">
				<div class="col-lg-12">
					<span class="close-btn"><i class="fas fa-window-close"></i></span>
					<div class="search-bar">
						<div class="search-bar-tablecell">
							<h3>Search For:</h3>
							<input type="text" placeholder="Keywords">
							<button type="submit">Search <i class="fas fa-search"></i></button>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
	<!-- end search arewa -->

	<!-- breadcrumb-section -->
	<div class="breadcrumb-section breadcrumb-bg">
		<div class="container">
			<div class="row">
				<div class="col-lg-8 offset-lg-2 text-center">
					<div class="breadcrumb-text">
						<p>Get 24/7 Support</p>
						<h1>Signup</h1>
					</div>
				</div>
			</div>
		</div>
	</div>
	<!-- end breadcrumb section -->

	<!-- contact form -->
	<div class="contact-from-section mt-150 mb-150">
		<div class="container">
			<div class="row">
				<div class="col-lg-8 mb-5 mb-lg-0">
					<div class="form-title">
						<h2>Registration Details</h2>
					</div>
					<div id="form_status"></div>
					<div class="contact-form">
						<form method="post" action="RegistrationController">
							<p>
							
								<input type="text" placeholder="First Name" name="firstname" id="firstname">
								<input type="text" placeholder="Last Name" name="lastname" id="lastname">						
							</p>
							
							
							
							<p>
							Gender
							<div class="form-check form-check-inline">
  <input class="form-check-input" type="radio" name="genderradio" id="inlineRadio1" value="Male">
  <label class="form-check-label" for="inlineRadio1">Male</label>
</div>
<div class="form-check form-check-inline">
  <input class="form-check-input" type="radio" name="genderradio" id="inlineRadio2" value="Female">
  <label class="form-check-label" for="inlineRadio2">Female</label>
</div>

							<p>
								<input style="margin-left:0" type="email" placeholder="Email" name="emailaddress" id="email">
								<input type="text" placeholder="Phone" name="phonenumber" id="phone">
							</p>
							
							
							
							
							
							
							<p>
							<textarea name="address" id="address" cols="30" rows="10"
									placeholder="Enter your address here"></textarea>
							</p>
							<p>
							<input type="text" placeholder="City" name="city" id="city">
							<input style="margin-right:0" type="text" placeholder="Pincode" name="pincode" id="pincode">
							</p>
							<input type="hidden" name="token" value="FsWga4&@f6aw" />
							
							
							<p><input style="padx:10px;line-height: 3.5; width: -webkit-fill-available" type="password" placeholder="Password" name="password" id="password">
							</p>
							<p><input type="submit" value="Sign Up"></p>
						</form>
					</div>
				</div>
				<div class="col-lg-4">
					<div class="contact-form-wrap">
						<div class="contact-form-box">
							<h4><i class="fas fa-map"></i> Shop Address</h4>
							<p>34/8, East Hukupara <br> Gifirtok, Sadan. <br> Country Name</p>
						</div>
						<div class="contact-form-box">
							<h4><i class="far fa-clock"></i> Shop Hours</h4>
							<p>MON - FRIDAY: 8 to 9 PM <br> SAT - SUN: 10 to 8 PM </p>
						</div>
						<div class="contact-form-box">
							<h4><i class="fas fa-address-book"></i> Contact</h4>
							<p>Phone: +00 111 222 3333 <br> Email: nearbyservices2023@gmail.com</p>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
	<!-- end contact form -->


	

<jsp:include page="footer.jsp" />





</body>

</html>