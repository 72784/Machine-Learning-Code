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
	<%@ include file="cust-header.jsp"%>
	<br>
	<br>
	<br>
	<br>
	<div class="row" style="margin-top: 53px">
		<div class="col-lg-8 offset-lg-2 text-center">
			<div class="section-title">
				<h3>
					<span class="orange-text">My</span> Orders
				</h3>
				<h1><%=request.getParameter("customerid")%></h1>
				<table class="table" style="margin-bottom: 421px; margin-top: 63px">
					<thead class="thead-light">
						<tr>
							<th scope="col">Booking ID</th>
							<th scope="col">Service Requirement</th>
							<th scope="col">Booking Date</th>
							<th scope="col">Status</th>
							<th scope="col">Cost</th>
							<th scope="col">Servicemen</th>
							<th scope="col">Service Date</th>
							<th scope="col">Service Time</th>
						</tr>
					</thead>

					<tbody>
						<%
						Connection cn = new DBUtil().getDatabaseConnection();
						PreparedStatement st = cn.prepareStatement("select * from book where userid=?");
						st.setString(1, request.getParameter("customerid"));
						ResultSet rs = st.executeQuery();
						while (rs.next()) {
						%>
						<tr>


							<td><%=rs.getString("bookid")%></td>
<%
							Connection cn3 = new DBUtil().getDatabaseConnection();
							PreparedStatement stthird = cn3.prepareStatement("select * from subservicei where id=?");
							stthird.setString(1, rs.getString("id"));
							ResultSet rs3 = stthird.executeQuery();
							while (rs3.next()) {
							%>
							<td><%=rs3.getString("description")%></td>

							<%
							}
							cn3.close();
							%>

							<td><%=rs.getString("bdate")%></td>
							<%
									if (rs.getString("status").equalsIgnoreCase("Contract Closed")) {
									%>
						<td><%=rs.getString("status")%>	<a
									href="feedback_form.jsp?customerid=<%=request.getParameter("customerid")%>"><button
										type="submit" 
										class="btn btn-outline-info">Feedback Form</button></a></td>
										<%}else{ %>
																	<td><%=rs.getString("status")%></td>
										<% }%>
							<td><%=rs.getString("price")%></td>
							<%
							Connection cn2 = new DBUtil().getDatabaseConnection();
							PreparedStatement st2 = cn2.prepareStatement("select * from assign where bookid=?");
							st2.setString(1, rs.getString("bookid"));

							ResultSet rs2 = st2.executeQuery();
							while (rs2.next()) {
							%>
							<td>
							<%
									if (rs.getString("status").equalsIgnoreCase("Contract Accepted")||rs.getString("status").equalsIgnoreCase("Contract Closed")) {
									%>
							<%=rs2.getString("servicemen")%></td>
							<td><%=rs2.getString("date")%></td>
							<td><%=rs2.getString("time")%></td>
							<%}else{ %>
							NA </td>
						<td>	NA </td>
						<td>	NA </td>
							<%} %>
							<%
							}
							cn2.close();
							%>

						</tr>
					

						<%
						}
						cn.close();
						%>


					</tbody>
				</table>
			</div>
		</div>
	</div>

	<jsp:include page="footer.jsp" />

</body>
</html>