<%@page import="java.sql.ResultSet"%>
<%@page import="java.sql.PreparedStatement"%>
<%@page import="com.util.DBUtil"%>
<%@page import="java.sql.Connection"%>
<%@ page language="java" contentType="text/html; charset=ISO-8859-1"
	pageEncoding="ISO-8859-1"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="ISO-8859-1">
<title>Insert title here</title>
<!-- Favicon icon -->
<link rel="icon" type="image/png" sizes="16x16"
	href="images/favicon.png">
<!-- Pignose Calender -->
<link href="./plugins/pg-calendar/css/pignose.calendar.min.css"
	rel="stylesheet">
<!-- Chartist -->
<link rel="stylesheet" href="./plugins/chartist/css/chartist.min.css">
<link rel="stylesheet"
	href="./plugins/chartist-plugin-tooltips/css/chartist-plugin-tooltip.css">
<!-- Custom Stylesheet -->
<link href="css/style.css" rel="stylesheet">

</head>
<body>
	<jsp:include page="header_sidebar.jsp" />


	<div class="content-body">
		<div class="card-body">
			<div class="active-member">
				<div class="table-responsive">
					<table class="table table-xs mb-0">
						<thead>
							<tr>
								<th>Book id</th>
								<th>Customers</th>
								<th>City</th>
								<th>Email</th>
								<th>Phone Number</th>
								<th>Service</th>
								<th>Price</th>
								<th>Status</th>
								<th>Payment Method</th>
								<th>Assignment</th>

							</tr>
						</thead>
						<tbody>

							<%
							Connection cn = new DBUtil().getDatabaseConnection();
							PreparedStatement st = cn.prepareStatement("select * from book");
							ResultSet rs = st.executeQuery();
							while (rs.next()) {
							%>
							<tr>
								<%
								Connection cn1 = new DBUtil().getDatabaseConnection();
								PreparedStatement st1 = cn1.prepareStatement("select * from customer where userid=?");
								st1.setString(1, rs.getString(3));
								ResultSet rs1 = st1.executeQuery();
								while (rs1.next()) {
								%>

								<td><%=rs.getString("bookid")%></td>

								<td><%=rs1.getString("firstname") + " " + rs1.getString("lastname")%></td>
								<td><%=rs1.getString("city")%></td>
								<td><%=rs1.getString("emailaddress")%></td>
								<td><%=rs1.getString("phonenumber")%></td>

								<%
								}
								cn1.close();
								%>

								<%
								Connection cn2 = new DBUtil().getDatabaseConnection();
								PreparedStatement st2 = cn2.prepareStatement("select * from subservicei where id=?");
								st2.setString(1, rs.getString(2));
								ResultSet rs2 = st2.executeQuery();
								while (rs2.next()) {
								%>
								<td><%=rs2.getString("title")%></td>



								<td><%=rs.getString("price")%></td>
								<td><%=rs.getString("status")%></td>
								<td><%=rs.getString("pstatus")%></td>
								<%
								Connection cn3 = new DBUtil().getDatabaseConnection();
								PreparedStatement st3 = cn3.prepareStatement("select * from mainservice where serviceid=?");
								st3.setString(1, rs2.getString(2));
								ResultSet rs3 = st3.executeQuery();
								while (rs3.next()) {
								%>
								<td>
									<%
									if (rs.getString("status").equalsIgnoreCase("Pending")||rs.getString("status").equalsIgnoreCase("Request canceled by admin")) {
									%> <a
									href="assign.jsp?bookid=<%=rs.getString("bookid")%>&serviceid=<%=rs3.getString(1)%>"><button
											type="submit" name="action" value="contract"
											class="btn btn-primary">Assign</button></a>

								</td>
								<%
								} else if (rs.getString("status").equalsIgnoreCase("Request denied by servicemen")) {
								%>
								<a
									href="assign.jsp?bookid=<%=rs.getString("bookid")%>&serviceid=<%=rs3.getString(1)%>"><button
										type="submit" name="action" value="contract"
										class="btn btn-danger">Assign Other Servicemen</button></a>
								</td>
								<%
								} else if (rs.getString("status").equalsIgnoreCase("Contract Accepted")) {
								%>
								<a
									href="#"><button
										type="submit" name="action" value="contract"
										class="btn btn-warning">Work in progress</button></a>
								</td>
								<%
								} else if (rs.getString("status").equalsIgnoreCase("Contract Closed")) {
								%>
								<a href="contract_details.jsp?bookid=<%=rs.getString("bookid")%>"><button
										type="submit" name="action" value="contract"
										class="btn btn-success">View Contract</button></a>
										
								</td>
								<%
								}else if (rs.getString("status").equalsIgnoreCase("Request sent by admin")) {
								%>
								<form method="post" action="../AdminController">
								<input type="hidden" name="bookid" value="<%=rs.getString("bookid") %>">
								<button
										type="submit" name="action" value="cancel-request-by-admin"
										class="btn btn-danger">Cancel Request</button>
							 </form>
								</td>
								<%} %>
								<%
								}
								cn3.close();
								%>

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
	</div>

	<jsp:include page="footer.jsp" />
	<!--**********************************
        Scripts
    ***********************************-->
	<script src="plugins/common/common.min.js"></script>
	<script src="js/custom.min.js"></script>
	<script src="js/settings.js"></script>
	<script src="js/gleek.js"></script>
	<script src="js/styleSwitcher.js"></script>

	<!-- Chartjs -->
	<script src="./plugins/chart.js/Chart.bundle.min.js"></script>
	<!-- Circle progress -->
	<script src="./plugins/circle-progress/circle-progress.min.js"></script>
	<!-- Datamap -->
	<script src="./plugins/d3v3/index.js"></script>
	<script src="./plugins/topojson/topojson.min.js"></script>
	<script src="./plugins/datamaps/datamaps.world.min.js"></script>
	<!-- Morrisjs -->
	<script src="./plugins/raphael/raphael.min.js"></script>
	<script src="./plugins/morris/morris.min.js"></script>
	<!-- Pignose Calender -->
	<script src="./plugins/moment/moment.min.js"></script>
	<script src="./plugins/pg-calendar/js/pignose.calendar.min.js"></script>
	<!-- ChartistJS -->
	<script src="./plugins/chartist/js/chartist.min.js"></script>
	<script
		src="./plugins/chartist-plugin-tooltips/js/chartist-plugin-tooltip.min.js"></script>



	<script src="./js/dashboard/dashboard-1.js"></script>


</body>
</html>