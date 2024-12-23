<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Quixlab - Bootstrap Admin Dashboard Template by Themefisher.com</title>
    <!-- Favicon icon -->
    <link rel="icon" type="image/png" sizes="16x16" href="images/favicon.png">
    <!-- Pignose Calender -->
    <link href="./plugins/pg-calendar/css/pignose.calendar.min.css" rel="stylesheet">
    <!-- Chartist -->
    <link rel="stylesheet" href="./plugins/chartist/css/chartist.min.css">
    <link rel="stylesheet" href="./plugins/chartist-plugin-tooltips/css/chartist-plugin-tooltip.css">
    <!-- Custom Stylesheet -->
    <link href="css/style.css" rel="stylesheet">

</head>

<body>

	<jsp:include page="header_sidebar.jsp" />
<div class="content-body">

             <div class="container-fluid">
                <div class="row">
                    <div class="col-lg-12">
                        <div class="card">
                            <div class="card-body">
                                <h4 class="card-title">Input Size</h4>
                                <p class="text-muted">Set heights using classes like <code>.form-control-lg and .form-control-sm.</code>
                                </p>
                                <div class="basic-form">
                                    <form method="post" action="../AdminController" enctype="multipart/form-data">
                                        <div class="form-group">
<label for="name" class="form-label">Service Name : </label>
                                        
                                            <input name="type" id="name" class="form-control form-control-lg" type="text" style="width: 600px; height: 30px" placeholder="Add Service name" fdprocessedid="u1yvs">
                                        </div>
                                       <div class="form-group">
 <label for="description" class="form-label">Service Description : </label>
                                       
                                            <input name="description" id="description" class="form-control form-control-lg" type="text" style="width: 600px; height: 160px" placeholder="Description" fdprocessedid="u1yvs">
                                        </div>
                                        <div class="form-group">
                                        <label for="image" class="form-label">Service Image : </label>
                                            <input name="image" id="image" class="form-control form-control-lg" type="file" style="width: 600px; height: 30px" >
                                        </div>
<button type="submit" name="action" value="main-service" class="btn mb-1 btn-rounded btn-primary btn-lg" fdprocessedid="4u2kq">Add New Services</button>                  
                  </form>
                                </div>
                            </div>
                        </div>
                    </div>   	
                    </div>   	
                    </div>   	
                    </div>
    <jsp:include page="footer.jsp" />
        
        
    <!--**********************************
        Main wrapper end
    ***********************************-->

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
    <script src="./plugins/chartist-plugin-tooltips/js/chartist-plugin-tooltip.min.js"></script>



    <script src="./js/dashboard/dashboard-1.js"></script>

</body>

</html>