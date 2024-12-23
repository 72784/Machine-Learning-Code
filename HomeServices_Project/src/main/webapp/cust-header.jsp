<%@page import="java.sql.ResultSet"%>
<%@page import="java.sql.PreparedStatement"%>
<%@page import="com.util.DBUtil"%>
<%@page import="java.sql.Connection"%>
<%@ page language="java" contentType="text/html; charset=ISO-8859-1"
    pageEncoding="ISO-8859-1"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="ISO-8859-1"><script>  
var request=new XMLHttpRequest();  
function searchInfo()
{  
	
	var name=document.vinform.name.value;  
	var url="AjaxSearch.jsp?val="+name;  
  	try
  	{  
		request.onreadystatechange=function()
		{  
			if(request.readyState==4)
			{  
				var val=request.responseText;  
				document.getElementById('tops').innerHTML=val;  
			}  
		}  
		request.open("GET",url,true);  
		request.send();  
	}
  	catch(e)
  	{
  		alert("Unable to connect to server");
  	}  
}  
</script>
<style type="text/css">
.links a{
color: white;
}
.links a:hover{
color: #F28123;
}
</style>
<title>Insert title here</title>

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
							<a href="afterLogin.jsp">
								<img src="assets/img/newlogo.png" style="height: 60px;width: 85px;" alt="No PICTURE">
							</a>
						</div>
						<!-- logo -->

						<!-- menu start -->
						<nav class="main-menu">
							<ul>
								

								<li><a href="about.html">About</a></li>

								<li><a href="login.jsp">Contact</a>
									<ul class="sub-menu">
										<li><a href="login.jsp">Login</a></li>
										<li><a href="signup.jsp">Signup</a></li>
									</ul>
								</li>

								<li><a href="services.jsp">Services</a>
								
								<li class="current-list-item"><a style="text-transform: uppercase" href="#">${model.firstname }</a>
								<ul class="sub-menu">
										<li><a  href="view-cust-profile.jsp?customerid=${model.customerid }">View Profile</a></li>
										<li><a href="myorders.jsp?customerid=${model.customerid }">My Orders</a></li>
									</ul>
								</li>
									
								
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
	<!-- search area -->
	 
	<div class="search-area">
		<div class="container">
			<div class="row">
				<div class="col-lg-12">
					<span class="close-btn"><i class="fas fa-window-close"></i></span>
					<div class="search-bar">
						<div class="search-bar-tablecell" style="vertical-align:top">
							<h3 style="margin-bottom:15px;margin-top: 80px;font-size: 15px">Search For:</h3>
						<form name="vinform" action="LoginController" method="post">
							<input name="name" type="text" name="servicename" onkeyup="searchInfo()" style="margin-bottom: 20px; font-size: 30px; padding: 0px"
									placeholder="Keywords" style="padding:0">>
							
							<span class="links" id="tops"></span>
						</form>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
   
  
	<!-- end search area -->

</body>
</html>