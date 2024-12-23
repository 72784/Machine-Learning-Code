<%@page import="com.util.DBUtil"%>
<%@page import="java.sql.ResultSet"%>
<%@page import="java.sql.PreparedStatement"%>
<%@page import="java.sql.Connection"%>
<%@ page language="java" contentType="text/html; charset=ISO-8859-1"
    pageEncoding="ISO-8859-1"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="ISO-8859-1">

<title>Insert title here</title>


<jsp:include page="styles.jsp" />

</head>
<body>
<%@ include file="cust-header.jsp" %>
	

		
		 
		<!-- product section -->
	<div class="product-section mt-150 mb-150">
		<div class="container">
			<div class="row">
				<div class="col-lg-8 offset-lg-2 text-center">
					<div class="section-title">
						<h3><span class="orange-text">Our</span> Services</h3>
						<p>"Expert solutions for your home repairs. Trust us for reliable service, quality workmanship, and peace of mind."</p>
					</div>
				</div>
			</div>
			<div class="row">
			

 
	                                        <%
                                             Connection cn=new DBUtil().getDatabaseConnection();
                                             PreparedStatement st = cn.prepareStatement("select * from mainservice");
                                         ResultSet rs=st.executeQuery();
                                         while(rs.next())
                                         {
                                             %>
				<div class="col-lg-4 col-md-6 text-center">
					<div class="single-product-item">
						<div class="product-image">
							<a href="viewsubservice.jsp?serviceid=<%=rs.getInt(1) %>"><img src="uploadimage/<%=rs.getString(4)%>" alt=""></a>
						</div>
						<h3><%= rs.getString(2) %></h3>
												<h3><%= rs.getString(3) %></h3>
						
						<a href="viewsubservice.jsp?serviceid=<%=rs.getInt(1) %>" class="cart-btn"><i class="fas fa-shopping-cart"></i>Show Sub Services </a>
					</div>
				</div>
				<%
                                         }
                                         cn.close();
				%>
					</div>
					</div>
					</div>
	<jsp:include page="footer.jsp" />
	

</body>
</html>