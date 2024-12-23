<!DOCTYPE html>
<%@page import="java.sql.ResultSet"%>
<%@page import="java.sql.PreparedStatement"%>
<%@page import="com.util.DBUtil"%>
<%@page import="java.sql.Connection"%>
<html lang="en">

<head>
<!-- Required meta tags -->
<meta charset="utf-8">
<meta name="viewport"
	content="width=device-width, initial-scale=1, shrink-to-fit=no">
<title>RoyalUI Admin</title>
<!-- plugins:css -->
<link rel="stylesheet" href="vendors/ti-icons/css/themify-icons.css">
<link rel="stylesheet" href="vendors/base/vendor.bundle.base.css">
<!-- endinject -->
<!-- plugin css for this page -->
<!-- End plugin css for this page -->
<!-- inject:css -->
<link rel="stylesheet" href="css/style.css">
<!-- endinject -->
<link rel="shortcut icon" href="images/favicon.png" />
</head>
<body>
	<jsp:include page="headers.jsp" />



	<div class="main-panel">
		<div class="content-wrapper" style="background-color: white">

			<h4 class="card-title display-4">Contract Details</h4>
			<div class="card-body">

				<form class="forms-sample" method="post" action="../AdminController">
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
					<div class="form-group row" style="margin-bottom: 0px">
						<label for="exampleInputUsername2" class="col-sm-3 col-form-label">Name</label>
						<div class="col-sm-9">
							<input readonly name="servicemen"
								value="<%=rs3.getString("firstname") + " " + rs3.getString("lastname")%>
										"
								type="text" class="form-control" id="exampleInputUsername2">
						</div>
					</div>
					<div class="form-group row" style="margin-bottom: 0px">
						<label for="exampleInputMobile" class="col-sm-3 col-form-label">Service
							Requirement</label>

						<div class="col-sm-9">
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
							stfourth.setString(1, rs1.getString("id"));
							ResultSet rs4 = stfourth.executeQuery();
							while (rs4.next()) {
							%>
							<input value="<%=rs4.getString("title")%>" readonly
								class="form-control" id="exampleInputMobile">


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
					<div class="form-group row" style="margin-bottom: 0px">
						<label for="exampleInputMobile" class="col-sm-3 col-form-label">Sub-Service
							Requirement</label>

						<div class="col-sm-9">
							<%
							Connection cn6 = new DBUtil().getDatabaseConnection();
							PreparedStatement st6 = cn6.prepareStatement("select * from book where bookid=?");
							st6.setInt(1, Integer.parseInt(request.getParameter("bookid")));
							ResultSet rs6 = st6.executeQuery();
							while (rs6.next()) {
							%>
							<%
							Connection cn7 = new DBUtil().getDatabaseConnection();
							PreparedStatement st7 = cn7.prepareStatement("select * from subservicei where id=?");
							st7.setString(1, rs6.getString("id"));
							ResultSet rs7 = st7.executeQuery();
							while (rs7.next()) {
							%>
							<input value="<%=rs7.getString("description")%>" readonly
								class="form-control" id="exampleInputMobile">


							<%
							}
							cn7.close();
							%>
							<%
							}
							cn6.close();
							%>
						</div>
					</div>

					<div class="form-group row" style="margin-bottom: 0px">
						<label for="exampleInputEmail2" class="col-sm-3 col-form-label">Email</label>
						<div class="col-sm-9">
							<input type="email" name="email" readonly
								value="<%=rs3.getString("emailaddress")%>
										"
								class="form-control" id="exampleInputEmail2">
						</div>
					</div>
					<div class="form-group row" style="margin-bottom: 0px">
						<label for="exampleInputMobile" class="col-sm-3 col-form-label">Mobile</label>
						<div class="col-sm-9">
							<input value="<%=rs3.getString("phonenumber")%>" readonly
								class="form-control" id="exampleInputMobile"">
						</div>
					</div>

					<div class="form-group row" style="margin-bottom: 0px">
						<label for="exampleInputMobile" class="col-sm-3 col-form-label">Address</label>
						<div class="col-sm-9">
							<input value="<%=rs3.getString("address")%>" readonly
								class="form-control" id="exampleInputMobile">
						</div>
					</div>
					<div class="form-group row" style="margin-bottom: 0px">
						<label for="exampleInputMobile" class="col-sm-3 col-form-label">City</label>
						<div class="col-sm-9">
							<input value="<%=rs3.getString("city")%>" readonly
								class="form-control" id="exampleInputMobile">
						</div>
					</div>
					<div class="form-group row" style="margin-bottom: 0px">
						<label for="exampleInputMobile" class="col-sm-3 col-form-label">Pincode</label>
						<div class="col-sm-9">
							<input value="<%=rs3.getString("pincode")%>" readonly
								class="form-control" id="exampleInputMobile">
						</div>
					</div>
					<%
					}

					cn3.close();
					%>
					<%
					}
					cn2.close();
					%>

					<%
					Connection cn5 = new DBUtil().getDatabaseConnection();
					PreparedStatement st5 = cn5.prepareStatement("select * from assign where bookid=?");
					st5.setString(1, request.getParameter("bookid"));
					ResultSet rs5 = st5.executeQuery();

					while (rs5.next()) {
					%>
					<div class="form-group row" style="margin-bottom: 0px">
						<label class="col-sm-3 col-form-label">Date</label>
						<div class="col-sm-9">
							<input name="date" value="<%=rs5.getString("date")%>" readonly
								class="form-control" id="exampleInputMobile">
						</div>

						<label class="col-sm-3 col-form-label">Time</label>
						<div class="col-sm-9">
							<input name="time" value="<%=rs5.getString("time")%>" readonly
								class="form-control" id="exampleInputMobile">
						</div>

					</div>

					<%
					}

					cn5.close();
					%>
					<div class="form-group row" style="margin-bottom: 0px">

						<label name="extra_cost" class="col-sm-3 col-form-label">Extra
							Service Cost</label>
						<div class="col-sm-9">
							<input name="extra_cost" placeholder="Extra Cost in INR"
								class="form-control" id="exampleInputMobile">
						</div>
					</div>


					<input type="hidden" name="bookid"
						value="<%=request.getParameter("bookid")%>"> <input
						type="hidden" name="email"
						value="<%=request.getParameter("email")%>">

					<button type="submit" name="action" value="completed"
						class="btn btn-primary mr-2">Contract Successfully
						Completed</button>
					<button class="btn btn-dark">Back</button>
				</form>
			</div>




		</div>


	</div>
	<jsp:include page="footer.jsp" />

</body>

</html>

