<%@page import="com.model.UserModel"%>
<%@ page language="java" contentType="text/html; charset=ISO-8859-1"
    pageEncoding="ISO-8859-1"%>
<!DOCTYPE html>
<html>

<head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>aranoz</title>
    <link rel="icon" href="img/favicon.png">
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="css/bootstrap.min.css">
    <!-- animate CSS -->
    <link rel="stylesheet" href="css/animate.css">
    <!-- owl carousel CSS -->
    <link rel="stylesheet" href="css/owl.carousel.min.css">
    <!-- font awesome CSS -->
    <link rel="stylesheet" href="css/all.css">
    <!-- flaticon CSS -->
    <link rel="stylesheet" href="css/flaticon.css">
    <link rel="stylesheet" href="css/themify-icons.css">
    <!-- font awesome CSS -->
    <link rel="stylesheet" href="css/magnific-popup.css">
    <!-- swiper CSS -->
    <link rel="stylesheet" href="css/slick.css">
    <!-- style CSS -->
    <link rel="stylesheet" href="css/style.css">
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
							<a href="index.jsp">
								<img src="assets/img/newlogo.png" alt="No PICTURE">
							</a>
						</div>
						<!-- logo -->

						<!-- menu start -->
						<nav class="main-menu">
							<ul>
								
<li class="current-list-item"><a style="text-transform: uppercase" href="#">Home</a>
								</li>
									
								<li><a href="about.html">About</a></li>

								<li><a href="login.jsp">Contact</a>
									<ul class="sub-menu">
										<li><a href="login.jsp">Login</a></li>
										<li><a href="signup.jsp">Signup</a></li>
									</ul>
								</li>

								<li><a href="services.jsp">Services</a></li>
								
								
								
								<li>
									<div class="header-icons">
										<a class="" href="cart.html"><i class="fas fa-shopping-cart"></i></a>
										<a class="mobile-hide search-bar-icon" href="#"><i
												class="fas fa-search"></i></a>
									</div>
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
	
	 <!-- breadcrumb start-->
    <section class="breadcrumb breadcrumb_bg">
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-lg-8">
                    <div class="breadcrumb_iner">
                        <div class="breadcrumb_iner_item">
                            <h2>Customer</h2>
                            <p>Home <span>-</span> Login Details</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>
    <!-- breadcrumb start-->
	
	${invalidemail }
	<section class="login_part padding_top">
        <div class="container">
            <div class="row align-items-center">
                
                <div class="col-lg-6 col-md-6">
                    <div class="login_part_form">
                        <div class="login_part_form_iner">
                            <h3>OTP Page</h3>
                            <%
String email=null;
if(session.getAttribute("otp")==null) 
{
response.sendRedirect("forgotpassword.jsp");
}
else	
{
	UserModel f=(UserModel)session.getAttribute("UserData");
	email=f.getEmailaddress();
}
%>
${notmatch }
                            <form class="row contact_form" action="ForgotController" method="post">
                                <div class="col-md-12 form-group p_star">
                                <label>Email Send On Email Address &emsp;<%=email%></label>
                                    <input type="text" class="form-control" id="name" name="EnterOTP"
                                        placeholder="Enter OTP">
                                </div>
                                
                                <div class="col-md-12 form-group">
                                     <div class="col-md-12 form-group">
                                    <button type="submit" name="action" value="verify" class="btn btn-primary btn-lg" style="padding-left: 2.5rem; padding-right: 2.5rem;" fdprocessedid="d538fo">Verify</button>
                                    </div>
                                    </div>
                              
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>
    <!--================login_part end =================-->
	
	
	
	
	<%@ include file="footer.jsp" %>


<!-- jquery plugins here-->
    <script src="js/jquery-1.12.1.min.js"></script>
    <!-- popper js -->
    <script src="js/popper.min.js"></script>
    <!-- bootstrap js -->
    <script src="js/bootstrap.min.js"></script>
    <!-- easing js -->
    <script src="js/jquery.magnific-popup.js"></script>
    <!-- swiper js -->
    <script src="js/swiper.min.js"></script>
    <!-- swiper js -->
    <script src="js/masonry.pkgd.js"></script>
    <!-- particles js -->
    <script src="js/owl.carousel.min.js"></script>
    <script src="js/jquery.nice-select.min.js"></script>
    <!-- slick js -->
    <script src="js/slick.min.js"></script>
    <script src="js/jquery.counterup.min.js"></script>
    <script src="js/waypoints.min.js"></script>
    <script src="js/contact.js"></script>
    <script src="js/jquery.ajaxchimp.min.js"></script>
    <script src="js/jquery.form.js"></script>
    <script src="js/jquery.validate.min.js"></script>
    <script src="js/mail-script.js"></script>
    <!-- custom js -->
    <script src="js/custom.js"></script>
</body>

</html>