<!DOCTYPE html>
<%@page import="com.model.BookModel"%>
<%@page import="java.sql.ResultSet"%>
<%@page import="java.sql.PreparedStatement"%>
<%@page import="com.util.DBUtil"%>
<%@page import="java.sql.Connection"%>
<%@page import="com.dao.LoginDao"%>
<%@page import="com.model.UserModel"%>
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

	

	<jsp:include page="cust-header.jsp" />

	

	<!-- breadcrumb-section -->
	<div class="breadcrumb-section breadcrumb-bg">
		<div class="container">
			<div class="row">
				<div class="col-lg-8 offset-lg-2 text-center">
					<div class="breadcrumb-text">
						<p>Get 24/7 Support</p>
						<h1>Customer Booking Details</h1>
					</div>
				</div>
			</div>
		</div>
	</div>
	<!-- end breadcrumb section -->
<%
UserModel model1 = (UserModel)session.getAttribute("model");

%>
	<!-- contact form -->
	<div class="contact-from-section mt-150 mb-150">
		<div class="container">
			<div class="row">
				<div class="col-lg-8 mb-5 mb-lg-0">
					<div class="form-title">
						<h2>Client Details</h2>
					</div>
					<div id="form_status"></div>
									<div class="contact-form">
					
    
						<form method="post" action="LoginController">
							<p>
							
								<input type="text" placeholder="First Name" name="firstname" id="firstname" value="<%=model1.getFirstname() %>">
								<input type="text" placeholder="Last Name" name="lastname" id="lastname" value="<%=model1.getLastname() %>">						
							</p>
							
							
							
								<input style="margin-left:0" type="email" placeholder="Email" name="emailaddress" id="email" value="<%=model1.getEmailaddress() %>">
								<input type="text" placeholder="Phone" name="phonenumber" id="phone" value="<%= model1.getPhonenumber() %>" >
							
							
							
							
							
							
							
							<p>
							<textarea name="address" id="address" cols="30" rows="10"
									placeholder="Enter your address here"> <%= model1.getAddress() %></textarea>
							</p>
							<p>
							<input type="text" placeholder="City" name="city" id="city" value="<%=model1.getCity() %>">
							<input style="margin-right:0" type="text" placeholder="Pincode" name="pincode" id="pincode" value="<%=model1.getPincode() %>">
							</p>
							
							<p>
							<input type="text" readonly="readonly" placeholder="City" name="city" id="city" value="<%=new LoginDao().getTitle(Integer.parseInt(request.getParameter("id"))) %>">
							<input type="hidden" name="id" value="<%=Integer.parseInt(request.getParameter("id")) %>">
							<input type="hidden" name="userid" value="<%= model1.getCustomerid()%>">
							
							<input style="margin-right:0" readonly="readonly" type="text" name="price" id="cost" value="<%=request.getParameter("cost") %>">
							</p>
							<input type="hidden" name="token" value="FsWga4&@f6aw" />
						<button name="action" type="submit" value="book order" class="btn btn-primary" style="
    background-color: #ff8700;
    border-color: white;
    color: white
"><i class="fas fa-shopping-cart"></i> Book order</button>
						
 <button type="button" class="btn btn-warning" style="
    background-color: #ff8700;
    border-color: white;
    color:white
">
               
                Cancel
              </button>							
								
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
							<p>Phone: +00 111 222 3333 <br> Email: support@fruitkha.com</p>
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