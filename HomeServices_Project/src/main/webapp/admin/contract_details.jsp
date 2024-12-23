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
			<div class="card-body">

				<h4 class="card-title">Details of the Contract</h4>
				<div class="basic-form">
					
						<%
						Connection cn2 = new DBUtil().getDatabaseConnection();
						PreparedStatement stsecond = cn2.prepareStatement("select * from book where bookid=?");
						stsecond.setString(1, request.getParameter("bookid"));
						ResultSet rs2 = stsecond.executeQuery();

						while (rs2.next()) {
						%>
						<%
						Connection cn3 = new DBUtil().getDatabaseConnection();
						PreparedStatement stthird = cn3.prepareStatement("select * from customer where userid=?");
						stthird.setString(1, rs2.getString("userid"));
						ResultSet rs3 = stthird.executeQuery();

						while (rs3.next()) {
						%>
						<div class="form-row">

							<div class="form-group col-md-6">
								<label>Customer Name</label> <input type="text" readonly
									value="<%=rs3.getString("firstname") + " " + rs3.getString("lastname")%>
										"
									class="form-control">
							</div>
							<div class="form-group col-md-6">
								<label>Service Requirement</label>
								<%
								Connection cn1 = new DBUtil().getDatabaseConnection();
								PreparedStatement stfirst = cn1.prepareStatement("select * from book where bookid=?");
								stfirst.setInt(1, Integer.parseInt(request.getParameter("bookid")));
								ResultSet rs1 = stfirst.executeQuery();
								while (rs1.next()) {
								%>
								<%
								Connection cn4 = new DBUtil().getDatabaseConnection();
								PreparedStatement stfourth = cn4.prepareStatement("select * from subservicei where id=?");
								stfourth.setString(1, rs1.getString(2));
								ResultSet rs4 = stfourth.executeQuery();
								while (rs4.next()) {
								%>
								<input type="text" value="<%=rs4.getString("title")%>" readonly
									class="form-control" placeholder="text">
								<%
								}
								cn4.close();
								%>
								<%
								}
								cn1.close();
								%>
							</div>
						</div>
						<div class="form-row">
							<div class="form-group col-md-6">
								<label>Email</label> <input type="email" readonly
									value="<%=rs3.getString("emailaddress")%>
										"
									class="form-control" placeholder="Email">
							</div>
							<div class="form-group col-md-6">
								<label>Mobile</label> <input
									value="<%=rs3.getString("phonenumber")%>" readonly type="text"
									class="form-control" placeholder="Password">
							</div>
						</div>
						<div class="form-group">
							<label>Address</label> <input
								value="<%=rs3.getString("address")%>" readonly type="text"
								class="form-control" placeholder="1234 Main St">
						</div>

						<div class="form-row">
							<div class="form-group col-md-6">
								<label>City</label> <input value="<%=rs3.getString("city")%>"
									readonly type="text" class="form-control">
							</div>
							<div class="form-group col-md-6">
								<label>pincode</label> <input
									value="<%=rs3.getString("pincode")%>" readonly type="text"
									class="form-control">
							</div>

						</div>
						<%
						}

						cn3.close();
						%>
						<%
						Connection cndt = new DBUtil().getDatabaseConnection();
						PreparedStatement stdt = cndt.prepareStatement("select * from assign where bookid=?");
						stdt.setInt(1, Integer.parseInt(request.getParameter("bookid")));
						ResultSet rsdt = stdt.executeQuery();
						while (rsdt.next()) {
						%>
						<div class="form-row">
							<div class="form-group col-md-6">
								<label>Date of completion</label> <input
									value="<%=rsdt.getString("date")%>" readonly type="text"
									class="form-control">
							</div>
							<div class="form-group col-md-6">
								<label>Time</label> <input value="<%=rsdt.getString("time")%>"
									readonly type="text" class="form-control">
							</div>

						</div>
						<%
						}
						cndt.close();
						%>

						<div class="form-group">
							<label>Extras</label> <input value="<%=rs2.getString("extracost")%>"
								readonly type="text" class="form-control"
								placeholder="1234 Main St">
						</div>
						
						<%
						}
						cn2.close();
						%>


						<a href="booking.jsp"><button 
								class="btn btn-dark">Back to bookings</button></a>

					

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