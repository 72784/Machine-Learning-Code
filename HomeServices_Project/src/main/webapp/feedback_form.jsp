<%@page import="java.sql.ResultSet"%>
<%@page import="java.sql.PreparedStatement"%>
<%@page import="com.util.DBUtil"%>
<%@page import="java.sql.Connection"%>
<%@page import="com.model.UserModel"%>
<%@ page language="java" contentType="text/html; charset=ISO-8859-1"
    pageEncoding="ISO-8859-1"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="ISO-8859-1">
<title>Insert title here</title>
<jsp:include page="styles.jsp" />
<jsp:include page="cust-header.jsp" />
<style type="text/css">
body{
		background-color: #25274d;
	}
	.contact{
		padding: 4%;
		height: 400px;
	}
	.col-md-3{
		background: #f28123;
		padding: 4%;
		border-top-left-radius: 0.5rem;
		border-bottom-left-radius: 0.5rem;
		margin-top:26px;
	}
	.contact-info{
		margin-top:10%;
	}
	.contact-info img{
		margin-bottom: 15%;
	}
	.contact-info h2{
		margin-bottom: 10%;
	}
	.col-md-9{
		background: #fff;
		padding: 3%;
		border-top-right-radius: 0.5rem;
		border-bottom-right-radius: 0.5rem;
	}
	.contact-form label{
		font-weight:600;
	}
	.contact-form button{
		background: #25274d;
		color: #fff;
		font-weight: 600;
		width: 25%;
	}
	.contact-form button:focus{
		box-shadow:none;
	}</style>
</head>
<body>
<link href="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
<script src="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/js/bootstrap.min.js"></script>
<script src="//cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
<!------ Include the above in your HEAD tag ---------->

<div>
<div style="margin-top: 105px; margin-bottom: 230px"
		class="product-section">
		<div class="container">
	<div class="row">
		<div class="col-md-3">
			<div class="contact-info">
				<img src="https://image.ibb.co/kUASdV/contact-image.png" alt="image"/>
				<h2>Contact Us</h2>
				<h4>We would love to hear from you !</h4>
			</div>
		</div>
		<form action="LoginController" method="post">
										<%
										Connection cn3 = new DBUtil().getDatabaseConnection();
										PreparedStatement stthird = cn3.prepareStatement("select * from customer where userid=?");
										stthird.setString(1, request.getParameter("customerid"));
										ResultSet rs3 = stthird.executeQuery();

										while (rs3.next()) {
										%>
		
		<div class="col-md-9" style="width:740px; margin-top: 44px">
			<div class="contact-form">
				<div class="form-group">
				  <label class="control-label col-sm-2" for="fname">Full Name</label>
				  <div class="col-sm-10">          
					<input readonly="readonly" type="text" value="<%=rs3.getString("firstname") + " " + rs3.getString("lastname")%>" class="form-control" id="fname" placeholder="Enter First Name" name="fullname">
				  </div>
				</div>
				
				</div>
				<div class="form-group">
				  <label class="control-label col-sm-2" for="email">Email:</label>
				  <div class="col-sm-10">
					<input type="email" readonly="readonly" class="form-control" value="<%=rs3.getString("emailaddress")%>" id="email" placeholder="Email" name="email">
				  </div>
				</div>
				<%} 
										
									
										cn3.close();
										%>
				
				<div class="form-group">
				  <label class="control-label col-sm-2" for="comment">Subject</label>
				  <div class="col-sm-10">
					<textarea class="form-control" rows="5" id="comment" name="subject" style="height:54px"></textarea>
				  </div>
				</div>
				<div class="form-group">
				  <label class="control-label col-sm-2" for="comment">Comment:</label>
				  <div class="col-sm-10">
					<textarea class="form-control" rows="5" id="comment" name="comment"></textarea>
				  </div>
				</div>
				
				<input type="hidden" name="userid" value="<%=request.getParameter("customerid") %>">
				
				<div class="form-group">        
				  <button type="submit" class="btn btn-primary btn-lg"
              style="padding-left: 2.5rem; padding-right: 2.5rem;margin-left: 97px;width: 235px" name="action" value="feedback">Submit feedback form</button>
            
				</div>
			</div>
			</form>
		</div>
		</div>
		
	
	</div>
</div>
<jsp:include page="footer.jsp" />

</body>
</html>