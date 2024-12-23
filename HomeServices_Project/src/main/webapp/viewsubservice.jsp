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

	
<jsp:include page="cust-header.jsp" />	
<!-- product section -->
	<div class="product-section mt-150 mb-150">
		<div class="container">
			<div class="row">
				<div class="col-lg-8 offset-lg-2 text-center">
					<div class="section-title">
						<h3><span class="orange-text">Sub</span> Services</h3>
						<p>Lorem ipsum dolor sit amet, consectetur adipisicing elit. Aliquid, fuga quas itaque eveniet
							beatae optio.</p>
					</div>
				</div>
			</div>
			<div class="row">
			
 <%
                                        Connection cn=new DBUtil().getDatabaseConnection();
                                        PreparedStatement st = cn.prepareStatement("select * from subservicei where parentid=?");
                                        st.setInt(1, Integer.parseInt(request.getParameter("serviceid")));
                                        ResultSet rs=st.executeQuery();
                                          while(rs.next())
                                         {
                                             %>
				<div class="col-lg-4 col-md-6 text-center">
					<div class="single-product-item">
						<div class="product-image">
							<a href="single-product.html"><img src="" alt=""></a>
						</div>
						<h3><%= rs.getString(3) %></h3>
						<h3><%= rs.getString(4) %></h3>
						
						<p class="product-price"> <span>Price</span> <h3>Rs.<%= rs.getString(5) %></h3> 
						</p>
						<a href="book.jsp?id=<%=rs.getInt(1) %>&cost=<%=rs.getString(5) %>" class="cart-btn"><i class="fas fa-shopping-cart"></i> Checkout </a>
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